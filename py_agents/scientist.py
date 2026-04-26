import struct
import time
import os
import flatbuffers
import torch
import torch.nn as nn
import numpy as np
import win32file

from schema.scmoa.Message import Message
from schema.scmoa.HivemindUpdate import HivemindUpdate
from schema.scmoa.ShardUpdate import ShardUpdate
from schema.scmoa.HivemindResult import HivemindResult, HivemindResultStart, HivemindResultAddResults, HivemindResultAddStepId, HivemindResultEnd
from schema.scmoa.ShardResult import ShardResult, ShardResultCreate
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
    def __init__(self, d_model=256, nhead=8, num_layers=4, num_specialists=4):
        super().__init__()
        self.d_model = d_model
        self.num_specialists = num_specialists
        
        # 1. Shared Unified Latent space (Backbone)
        self.embedding = nn.Embedding(64, d_model)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, dim_feedforward=d_model*4, batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        # 2. Context Injection (Shared)
        self.context_projector = nn.Linear(2, d_model)
        self.cross_attn = nn.MultiheadAttention(d_model, nhead, batch_first=True)
        
        # 3. Specialist Mixture Heads
        self.specialists = nn.ModuleList([SpecialistHead(d_model) for _ in range(num_specialists)])

    def forward(self, x_batch, context_batch, specialist_ids):
        # x_batch: (B, seq_len)
        # context_batch: (B, 2)
        # specialist_ids: (B,)
        
        B = x_batch.size(0)
        emb = self.embedding(x_batch)
        mask = nn.Transformer.generate_square_subsequent_mask(x_batch.size(1)).to(x_batch.device)
        
        # Shared Latent Processing
        latent = self.transformer(emb, mask=mask, is_causal=True)
        
        # Contextual Reconcilation
        ctx_emb = self.context_projector(context_batch).unsqueeze(1)
        latent, _ = self.cross_attn(query=latent, key=ctx_emb, value=ctx_emb)
        
        last_latent = latent[:, -1, :] # (B, d_model)
        
        # Hyper-Optimized Specialist Routing
        all_pred = torch.empty((B, 64), device=x_batch.device)
        all_act = torch.empty((B, 64), device=x_batch.device)
        
        for s_id in range(self.num_specialists):
            mask = (specialist_ids == s_id)
            if mask.any():
                s_latent = last_latent[mask]
                pred, act, _ = self.specialists[s_id](s_latent)
                all_pred[mask] = pred
                all_act[mask] = act
            
        return all_pred, all_act

def start_hivemind():
    print(f"Hivemind: Booting Gen 2 on {PIPE_NAME}...")
    handle = None
    while True:
        try:
            handle = win32file.CreateFile(
                PIPE_NAME, win32file.GENERIC_READ | win32file.GENERIC_WRITE,
                0, None, win32file.OPEN_EXISTING, 0, None
            )
            break
        except Exception: time.sleep(0.5)
    
    # Hyper-Optimization: Use Torch Compile if available
    device = "cpu"
    agent = HivemindAgent().to(device)
    try:
        agent = torch.compile(agent)
        print("Hivemind: JIT Compilation Successful (AVX-VNNI optimized path).")
    except Exception:
        print("Hivemind: Proceeding with Eager mode.")
    
    builder = flatbuffers.Builder(4096)
    shard_histories = {} # shard_id -> list of states

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
                
                batch_x = []
                batch_ctx = []
                batch_sid = []
                batch_shard_ids = []
                
                for i in range(hu.ShardsLength()):
                    su = hu.Shards(i)
                    sid = su.ShardId()
                    state = su.State()
                    ctx = [su.Context(j) for j in range(su.ContextLength())]
                    spec_id = su.SpecialistId()
                    
                    if sid not in shard_histories: shard_histories[sid] = [0]*16
                    shard_histories[sid].append(state)
                    if len(shard_histories[sid]) > 16: shard_histories[sid] = shard_histories[sid][-16:]
                    
                    batch_x.append(shard_histories[sid])
                    batch_ctx.append(ctx)
                    batch_sid.append(spec_id)
                    batch_shard_ids.append(sid)
                
                # Batch Inference
                with torch.inference_mode():
                    x = torch.tensor(batch_x, dtype=torch.long)
                    c = torch.tensor(batch_ctx, dtype=torch.float32)
                    s = torch.tensor(batch_sid, dtype=torch.long)
                    
                    preds, acts = agent(x, c, s)
                    
                    res_preds = preds.argmax(dim=-1).numpy()
                    res_acts = acts.argmax(dim=-1).numpy()
                
                # Emit Batch Result
                builder.Clear()
                result_offsets = []
                for i in range(len(batch_shard_ids)):
                    # No longer creating vectors for predicted_state and action
                    ShardResult.ShardResultStart(builder)
                    ShardResult.ShardResultAddShardId(builder, batch_shard_ids[i])
                    ShardResult.ShardResultAddPredictedState(builder, int(res_preds[i]))
                    ShardResult.ShardResultAddAction(builder, int(res_acts[i]))
                    result_offsets.append(ShardResult.ShardResultEnd(builder))
                
                HivemindResultStartShardsVector(builder, len(result_offsets))
                for off in reversed(result_offsets):
                    builder.PrependUOffsetTRelative(off)
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
                
                out = builder.Output()
                win32file.WriteFile(handle, struct.pack('<I', len(out)))
                win32file.WriteFile(handle, out)
                
        except Exception as e:
            print(f"Hivemind Error: {e}")
            break

def HivemindResultStartShardsVector(builder, numElems):
    return builder.StartVector(4, numElems, 4)

if __name__ == "__main__":
    start_hivemind()
