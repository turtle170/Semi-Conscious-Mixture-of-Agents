import struct
import time
import os
import flatbuffers
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import win32file
from safetensors.torch import save_file

from schema.scmoa.Message import Message
from schema.scmoa import HivemindUpdate, ShardUpdate, HivemindResult, ShardResult, Config, Checkpoint, Payload, StateUpdate

PIPE_NAME = r'\\.\pipe\scmoa_scientist'

class SpecialistHead(nn.Module):
    def __init__(self, d_model):
        super().__init__()
        self.prediction_head = nn.Linear(d_model, 64)
        self.action_head = nn.Linear(d_model, 64)
        self.value_head = nn.Linear(d_model, 1)
    def forward(self, x):
        return self.prediction_head(x), self.action_head(x), self.value_head(x)

class HivemindAgent(nn.Module):
    def __init__(self, d_model=128, nhead=8, num_layers=4, num_specialists=4):
        super().__init__()
        self.d_model, self.num_specialists = d_model, num_specialists
        self.embedding = nn.Embedding(64, d_model)
        el = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead, dim_feedforward=d_model*4, batch_first=True)
        self.transformer = nn.TransformerEncoder(el, num_layers=num_layers)
        self.context_projector = nn.Linear(2, d_model)
        self.cross_attn = nn.MultiheadAttention(d_model, nhead, batch_first=True)
        self.specialists = nn.ModuleList([SpecialistHead(d_model) for _ in range(num_specialists)])

    def forward(self, x_batch, context_batch, specialist_ids):
        B = x_batch.size(0)
        emb = self.embedding(x_batch)
        mask = nn.Transformer.generate_square_subsequent_mask(x_batch.size(1)).to(x_batch.device)
        latent = self.transformer(emb, mask=mask, is_causal=True)
        ctx_emb = self.context_projector(context_batch).unsqueeze(1)
        latent, _ = self.cross_attn(query=latent, key=ctx_emb, value=ctx_emb)
        last_latent = latent[:, -1, :]
        
        all_pred, all_act = torch.empty((B, 64), device=x_batch.device), torch.empty((B, 64), device=x_batch.device)
        for s_id in range(self.num_specialists):
            m = (specialist_ids == s_id)
            if m.any():
                p, a, _ = self.specialists[s_id](last_latent[m])
                all_pred[m], all_act[m] = p, a
        return all_pred, all_act

    def save_to_format(self, filepath, fmt, quant, max_seq):
        print(f"Agent: Exporting to {fmt} (Quant: {quant}) -> {filepath}")
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        sd = self.state_dict()
        if fmt.lower() == "pytorch":
            torch.save(sd, filepath)
        elif fmt.lower() == "safetensors":
            save_file(sd, filepath)
        elif fmt.lower() == "onnx":
            dummy_x = torch.zeros((1, max_seq), dtype=torch.long)
            dummy_c = torch.zeros((1, 2), dtype=torch.float32)
            dummy_s = torch.zeros((1,), dtype=torch.long)
            torch.onnx.export(self, (dummy_x, dummy_c, dummy_s), filepath, 
                              input_names=['states', 'context', 'specialists'],
                              output_names=['preds', 'actions'])
        else:
            print(f"Agent Warning: {fmt} export not natively implemented. Falling back to SafeTensors.")
            save_file(sd, filepath + ".safetensors")

def start_hivemind():
    print(f"Hivemind: Waiting on {PIPE_NAME}...")
    handle = None
    while True:
        try:
            handle = win32file.CreateFile(PIPE_NAME, win32file.GENERIC_READ | win32file.GENERIC_WRITE, 0, None, win32file.OPEN_EXISTING, 0, None)
            break
        except Exception: time.sleep(0.5)
    
    print("Hivemind: Synchronizing configuration...")
    err, lb = win32file.ReadFile(handle, 4)
    mlen = struct.unpack('<I', lb)[0]
    err, mbytes = win32file.ReadFile(handle, mlen)
    msg = Message.GetRootAsMessage(mbytes, 0)
    
    cfg = Config.Config()
    cfg.Init(msg.Payload().Bytes, msg.Payload().Pos)
    
    agent = HivemindAgent(d_model=cfg.HiddenDim(), nhead=cfg.Nhead(), num_layers=cfg.NumLayers(), num_specialists=cfg.NumSpecialists())
    out_fmt = cfg.OutputFormat().decode('utf-8')
    quant_type = cfg.Quantization().decode('utf-8')
    max_seq = cfg.MaxSeq()

    builder = flatbuffers.Builder(4096)
    shard_histories = {}

    while True:
        try:
            err, lb = win32file.ReadFile(handle, 4)
            if not lb: break
            mlen = struct.unpack('<I', lb)[0]
            err, mbytes = win32file.ReadFile(handle, mlen)
            msg = Message.GetRootAsMessage(mbytes, 0)
            
            p_type = msg.PayloadType()
            
            if p_type == Payload.Payload().HivemindUpdate:
                hu = HivemindUpdate.HivemindUpdate()
                hu.Init(msg.Payload().Bytes, msg.Payload().Pos)
                bx, bc, bs, bids = [], [], [], []
                for i in range(hu.ShardsLength()):
                    su = hu.Shards(i)
                    sid, state, spec_id = su.ShardId(), su.State(), su.SpecialistId()
                    ctx = [su.Context(j) for j in range(su.ContextLength())]
                    if sid not in shard_histories: shard_histories[sid] = [0]*max_seq
                    shard_histories[sid].append(state)
                    if len(shard_histories[sid]) > max_seq: shard_histories[sid] = shard_histories[sid][-max_seq:]
                    bx.append(shard_histories[sid]); bc.append(ctx); bs.append(spec_id); bids.append(sid)
                
                with torch.inference_mode():
                    x, c, s = torch.tensor(bx), torch.tensor(bc), torch.tensor(bs)
                    preds, acts = agent(x, c, s)
                    rp, ra = preds.argmax(dim=-1).numpy(), acts.argmax(dim=-1).numpy()
                
                builder.Clear()
                offsets = []
                for i in range(len(bids)):
                    ShardResult.ShardResultStart(builder)
                    ShardResult.ShardResultAddShardId(builder, bids[i])
                    ShardResult.ShardResultAddPredictedState(builder, int(rp[i]))
                    ShardResult.ShardResultAddAction(builder, int(ra[i]))
                    offsets.append(ShardResult.ShardResultEnd(builder))
                
                HivemindResult.HivemindResultStartResultsVector(builder, len(offsets))
                for o in reversed(offsets): builder.PrependUOffsetTRelative(o)
                rv = builder.EndVector()
                HivemindResult.HivemindResultStart(builder)
                HivemindResult.HivemindResultAddResults(builder, rv)
                HivemindResult.HivemindResultAddStepId(builder, hu.StepId())
                hr = HivemindResult.HivemindResultEnd(builder)
                
                from schema.scmoa.Message import MessageStart, MessageAddPayloadType, MessageAddPayload, MessageEnd
                MessageStart(builder); MessageAddPayloadType(builder, Payload.Payload().HivemindResult); MessageAddPayload(builder, hr)
                builder.Finish(MessageEnd(builder))
                win32file.WriteFile(handle, struct.pack('<I', len(builder.Output())))
                win32file.WriteFile(handle, builder.Output())

            elif p_type == Payload.Payload().Checkpoint:
                cp = Checkpoint.Checkpoint()
                cp.Init(msg.Payload().Bytes, msg.Payload().Pos)
                agent.save_to_format(cp.Filepath().decode('utf-8'), out_fmt, quant_type, max_seq)

        except Exception as e:
            print(f"Agent Error: {e}"); import traceback; traceback.print_exc(); break

if __name__ == "__main__":
    start_hivemind()
