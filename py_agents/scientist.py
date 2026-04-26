import struct
import time
import os
import flatbuffers
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import win32file

from schema.scmoa.Message import Message
from schema.scmoa.HivemindUpdate import HivemindUpdate
from schema.scmoa.ShardUpdate import ShardUpdate
from schema.scmoa.HivemindResult import HivemindResult, HivemindResultStart, HivemindResultAddResults, HivemindResultAddStepId, HivemindResultEnd
from schema.scmoa.ShardResult import ShardResult, ShardResultCreate
from schema.scmoa.Config import Config
from schema.scmoa.Payload import Payload

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
        self.d_model = d_model
        self.num_specialists = num_specialists
        self.embedding = nn.Embedding(64, d_model)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, dim_feedforward=d_model*4, batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
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
        
        all_pred = torch.empty((B, 64), device=x_batch.device)
        all_act = torch.empty((B, 64), device=x_batch.device)
        for s_id in range(self.num_specialists):
            mask_s = (specialist_ids == s_id)
            if mask_s.any():
                s_latent = last_latent[mask_s]
                pred, act, _ = self.specialists[s_id](s_latent)
                all_pred[mask_s] = pred
                all_act[mask_s] = act
        return all_pred, all_act

def start_hivemind():
    print(f"Hivemind: Booting SCMoA Agent on {PIPE_NAME}...")
    handle = None
    while True:
        try:
            handle = win32file.CreateFile(PIPE_NAME, win32file.GENERIC_READ | win32file.GENERIC_WRITE, 0, None, win32file.OPEN_EXISTING, 0, None)
            break
        except Exception: time.sleep(0.5)
    
    # 1. Synchronization: Wait for Config
    print("Hivemind: Synchronizing configuration...")
    err, len_bytes = win32file.ReadFile(handle, 4)
    msg_len = struct.unpack('<I', len_bytes)[0]
    err, msg_bytes = win32file.ReadFile(handle, msg_len)
    msg = Message.GetRootAsMessage(msg_bytes, 0)
    
    if msg.PayloadType() != Payload().Config:
        print("Hivemind Error: Expected Config message during sync.")
        return
        
    cfg = Config()
    cfg.Init(msg.Payload().Bytes, msg.Payload().Pos)
    
    # Initialize Agent with Configured Hyperparameters
    agent = HivemindAgent(
        d_model=cfg.HiddenDim(),
        nhead=cfg.Nhead(),
        num_layers=cfg.NumLayers(),
        num_specialists=cfg.NumSpecialists()
    )
    optimizer = optim.Adam(agent.parameters(), lr=cfg.LearningRate())
    max_seq = cfg.MaxSeq()
    
    print(f"Hivemind: Configured. LR={cfg.LearningRate()}, Dim={cfg.HiddenDim()}, Shards={cfg.BatchSize()}")

    try:
        agent = torch.compile(agent)
    except Exception: pass
    
    builder = flatbuffers.Builder(4096)
    shard_histories = {}

    while True:
        try:
            err, len_bytes = win32file.ReadFile(handle, 4)
            if not len_bytes: break
            msg_len = struct.unpack('<I', len_bytes)[0]
            err, msg_bytes = win32file.ReadFile(handle, msg_len)
            msg = Message.GetRootAsMessage(msg_bytes, 0)
            
            if msg.PayloadType() == Payload().HivemindUpdate:
                hu = HivemindUpdate()
                hu.Init(msg.Payload().Bytes, msg.Payload().Pos)
                step_id = hu.StepId()
                
                batch_x, batch_ctx, batch_sid, batch_shard_ids = [], [], [], []
                for i in range(hu.ShardsLength()):
                    su = hu.Shards(i)
                    sid, state, spec_id = su.ShardId(), su.State(), su.SpecialistId()
                    ctx = [su.Context(j) for j in range(su.ContextLength())]
                    
                    if sid not in shard_histories: shard_histories[sid] = [0]*max_seq
                    shard_histories[sid].append(state)
                    if len(shard_histories[sid]) > max_seq: shard_histories[sid] = shard_histories[sid][-max_seq:]
                    
                    batch_x.append(shard_histories[sid])
                    batch_ctx.append(ctx)
                    batch_sid.append(spec_id)
                    batch_shard_ids.append(sid)
                
                with torch.inference_mode():
                    x = torch.tensor(batch_x, dtype=torch.long)
                    c = torch.tensor(batch_ctx, dtype=torch.float32)
                    s = torch.tensor(batch_sid, dtype=torch.long)
                    preds, acts = agent(x, c, s)
                    res_preds = preds.argmax(dim=-1).numpy()
                    res_acts = acts.argmax(dim=-1).numpy()
                
                builder.Clear()
                result_offsets = []
                for i in range(len(batch_shard_ids)):
                    ShardResult.ShardResultStart(builder)
                    ShardResult.ShardResultAddShardId(builder, batch_shard_ids[i])
                    ShardResult.ShardResultAddPredictedState(builder, int(res_preds[i]))
                    ShardResult.ShardResultAddAction(builder, int(res_acts[i]))
                    result_offsets.append(ShardResult.ShardResultEnd(builder))
                
                HivemindResultStartShardsVector(builder, len(result_offsets))
                for off in reversed(result_offsets): builder.PrependUOffsetTRelative(off)
                res_vec = builder.EndVector()
                
                HivemindResultStart(builder)
                HivemindResultAddResults(builder, res_vec)
                HivemindResultAddStepId(builder, step_id)
                hr_root = HivemindResultEnd(builder)
                
                from schema.scmoa.Message import MessageStart, MessageAddPayloadType, MessageAddPayload, MessageEnd
                MessageStart(builder)
                MessageAddPayloadType(builder, Payload().HivemindResult)
                MessageAddPayload(builder, hr_root)
                builder.Finish(MessageEnd(builder))
                
                win32file.WriteFile(handle, struct.pack('<I', len(builder.Output())))
                win32file.WriteFile(handle, builder.Output())
                
        except Exception as e:
            print(f"Hivemind Agent Error: {e}")
            break

def HivemindResultStartShardsVector(builder, numElems):
    return builder.StartVector(4, numElems, 4)

if __name__ == "__main__":
    start_hivemind()
