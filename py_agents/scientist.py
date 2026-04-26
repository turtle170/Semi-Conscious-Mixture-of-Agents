import struct
import time
import os
import flatbuffers
import torch
import torch.nn as nn
import win32file
import win32pipe

from schema.scmoa.Message import Message
from schema.scmoa.StateUpdate import StateUpdate
from schema.scmoa.TopologyMutation import TopologyMutation
from schema.scmoa.Checkpoint import Checkpoint
from schema.scmoa.Prediction import PredictionStart, PredictionAddPredictedState, PredictionAddStepId, PredictionEnd
from schema.scmoa.Payload import Payload

PIPE_NAME = r'\\.\pipe\scmoa_scientist'

class WorldModel(nn.Module):
    def __init__(self, d_model=128, nhead=4, num_layers=2):
        super().__init__()
        self.embedding = nn.Embedding(num_embeddings=64, embedding_dim=d_model)
        encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead, dim_feedforward=d_model*4, batch_first=True)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.context_projector = nn.Linear(2, d_model)
        self.cross_attn = nn.MultiheadAttention(d_model, nhead, batch_first=True)
        self.output_head = nn.Linear(d_model, 64)
        self.d_model = d_model
        self.nhead = nhead
        self.num_layers = num_layers
        
    def forward(self, x, context_vec):
        emb = self.embedding(x)
        mask = nn.Transformer.generate_square_subsequent_mask(x.size(1)).to(x.device)
        self_out = self.transformer(emb, mask=mask, is_causal=True)
        ctx_emb = self.context_projector(context_vec).unsqueeze(1)
        attn_out, _ = self.cross_attn(query=self_out, key=ctx_emb, value=ctx_emb)
        logits = self.output_head(attn_out)
        return logits

    def save_checkpoint(self, filepath):
        print(f"Scientist: Saving checkpoint to {filepath}")
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        torch.save({
            'state_dict': self.state_dict(),
            'topology': {
                'd_model': self.d_model,
                'nhead': self.nhead,
                'num_layers': self.num_layers
            }
        }, filepath)

def start_scientist():
    print(f"Scientist: Waiting to connect to pipe {PIPE_NAME}...")
    handle = None
    while True:
        try:
            handle = win32file.CreateFile(PIPE_NAME, win32file.GENERIC_READ | win32file.GENERIC_WRITE, 0, None, win32file.OPEN_EXISTING, 0, None)
            break
        except Exception: time.sleep(0.5)
            
    print("Scientist: Connected to Distiller!")
    builder = flatbuffers.Builder(1024)
    model = WorldModel()
    model.eval()
    context_window = []

    # Optimization: pre-compute references outside loop
    payload_state_update = Payload().StateUpdate
    payload_checkpoint = Payload().Checkpoint
    payload_prediction = Payload().Prediction

    from schema.scmoa.Prediction import PredictionStart, PredictionAddPredictedState, PredictionAddStepId, PredictionEnd
    from schema.scmoa.Message import MessageStart, MessageAddPayloadType, MessageAddPayload, MessageEnd

    while True:
        try:
            err, len_bytes = win32file.ReadFile(handle, 4)
            if not len_bytes: break
            msg_len = struct.unpack('<I', len_bytes)[0]
            err, msg_bytes = win32file.ReadFile(handle, msg_len)
            msg = Message.GetRootAsMessage(msg_bytes, 0)
            
            payload_type = msg.PayloadType()

            if payload_type == payload_state_update:
                su = StateUpdate()
                su.Init(msg.Payload().Bytes, msg.Payload().Pos)

                # Optimized list comprehension
                state_arr = [su.State(i) for i in range(su.StateLength())]
                ctx_arr = [su.Context(i) for i in range(su.ContextLength())]

                context_window.extend(state_arr)
                if len(context_window) > 16: context_window = context_window[-16:]
                
                with torch.no_grad():
                    x = torch.tensor([context_window], dtype=torch.long)
                    ctx = torch.tensor([ctx_arr], dtype=torch.float32)
                    logits = model(x, ctx)
                    pred_token = logits[0, -1, :].argmax().item()
                
                builder.Clear()

                PredictionStartPredictedStateVector(builder, 1)
                builder.PrependUint8(pred_token)
                pred_vec = builder.EndVector()

                PredictionStart(builder)
                PredictionAddPredictedState(builder, pred_vec)
                PredictionAddStepId(builder, su.StepId())
                pred_table = PredictionEnd(builder)

                MessageStart(builder)
                MessageAddPayloadType(builder, payload_prediction)
                MessageAddPayload(builder, pred_table)
                builder.Finish(MessageEnd(builder))
                
                out_buf = builder.Output()
                win32file.WriteFile(handle, struct.pack('<I', len(out_buf)))
                win32file.WriteFile(handle, out_buf)

            elif payload_type == payload_checkpoint:
                cp = Checkpoint()
                cp.Init(msg.Payload().Bytes, msg.Payload().Pos)
                model.save_checkpoint(cp.Filepath().decode('utf-8'))
                
        except Exception as e:
            print(f"Scientist: Error: {e}")
            break

def PredictionStartPredictedStateVector(builder, numElems):
    return builder.StartVector(1, numElems, 1)

if __name__ == "__main__":
    start_scientist()
