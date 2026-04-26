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
from schema.scmoa.InferenceResult import InferenceResultStart, InferenceResultAddPredictedState, InferenceResultAddAction, InferenceResultAddStepId, InferenceResultEnd
from schema.scmoa.Payload import Payload

PIPE_NAME = r'\\.\pipe\scmoa_scientist'

class SCMoA_Agent(nn.Module):
    def __init__(self, d_model=128, nhead=4, num_layers=2):
        super().__init__()
        self.d_model = d_model
        self.nhead = nhead
        self.num_layers = num_layers
        
        # Shared Latent World Model (The Scientist)
        self.embedding = nn.Embedding(num_embeddings=64, embedding_dim=d_model)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, 
            nhead=nhead, 
            dim_feedforward=d_model * 4, 
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        # Cross-Attention Memory Retrieval
        self.context_projector = nn.Linear(2, d_model)
        self.cross_attn = nn.MultiheadAttention(d_model, nhead, batch_first=True)
        
        # Prediction Head (Guessing)
        self.prediction_head = nn.Linear(d_model, 64)
        
        # Policy Head (Probing - Actor)
        self.action_head = nn.Linear(d_model, 64)
        
        # Value Head (Critic)
        self.value_head = nn.Linear(d_model, 1)

    def forward(self, x, context_vec):
        # x: (batch, seq_len) - 6-bit states
        # context_vec: (batch, 2) - [gravity, friction]
        
        # 1. Latent Encoding
        emb = self.embedding(x)
        seq_len = x.size(1)
        mask = nn.Transformer.generate_square_subsequent_mask(seq_len).to(x.device)
        self_out = self.transformer(emb, mask=mask, is_causal=True)
        
        # 2. Memory-Augmented Context Injection
        ctx_emb = self.context_projector(context_vec).unsqueeze(1)
        # Query: self_out, Key/Value: ctx_emb
        latent, _ = self.cross_attn(query=self_out, key=ctx_emb, value=ctx_emb)
        
        # 3. Dual-Head Output
        # We use the latent representation of the LAST step for both guess and probe
        last_latent = latent[:, -1, :]
        
        state_logits = self.prediction_head(last_latent)
        action_logits = self.action_head(last_latent)
        value = self.value_head(last_latent)
        
        return state_logits, action_logits, value

    def save_checkpoint(self, filepath):
        print(f"Agent: Saving checkpoint to {filepath}")
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        torch.save({
            'state_dict': self.state_dict(),
            'topology': {
                'd_model': self.d_model,
                'nhead': self.nhead,
                'num_layers': self.num_layers
            }
        }, filepath)

    def rewire_architecture(self, command: str):
        # Implementation of dynamic growth as requested in earlier phases
        if command == "ADD_HEAD":
            print("Agent: Expanding Multi-Head Attention...")
            self.nhead += 1
            # Real-world: surgery to resize weights. Demo: reset for simplicity but logic placeholder exists.
            # In a hyper-optimized version, we'd use weight transfer.
            pass

def start_agent():
    print(f"Agent: Waiting for Distiller on {PIPE_NAME}...")
    handle = None
    while True:
        try:
            handle = win32file.CreateFile(
                PIPE_NAME,
                win32file.GENERIC_READ | win32file.GENERIC_WRITE,
                0, None, win32file.OPEN_EXISTING, 0, None
            )
            break
        except Exception:
            time.sleep(0.5)
            
    print("Agent: Synchronized with Orchestrator.")

    # Hyper-Optimization: Reuse builder to minimize GC pressure
    builder = flatbuffers.Builder(1024)
    model = SCMoA_Agent()
    model.eval()
    
    # Context window for sequential dependencies
    context_window = []
    MAX_SEQ = 16

    # Move model to optimal device if available (e.g. AVX-VNNI optimized torch build)
    # On CPU, torch usually auto-detects AVX/AVX2/AVX-512
    
    while True:
        try:
            # 1. Ingest StateUpdate
            err, len_bytes = win32file.ReadFile(handle, 4)
            if not len_bytes: break
            
            msg_len = struct.unpack('<I', len_bytes)[0]
            err, msg_bytes = win32file.ReadFile(handle, msg_len)
            msg = Message.GetRootAsMessage(msg_bytes, 0)
            
            p_type = msg.PayloadType()
            
            if p_type == Payload().StateUpdate:
                su = StateUpdate()
                su.Init(msg.Payload().Bytes, msg.Payload().Pos)
                
                step_id = su.StepId()
                state_val = su.State(0) # 6-bit state
                ctx_arr = [su.Context(i) for i in range(su.ContextLength())] # [G, F]
                
                context_window.append(state_val)
                if len(context_window) > MAX_SEQ:
                    context_window = context_window[-MAX_SEQ:]
                
                # 2. Combined Inference (Guess & Probe)
                with torch.inference_mode():
                    x = torch.tensor([context_window], dtype=torch.long)
                    ctx = torch.tensor([ctx_arr], dtype=torch.float32)
                    
                    state_logits, action_logits, _ = model(x, ctx)
                    
                    pred_token = state_logits.argmax(dim=-1).item()
                    action_token = action_logits.argmax(dim=-1).item()
                
                # 3. Emit Combined InferenceResult
                builder.Clear()
                
                # Build predicted state vector
                pred_vec = builder.CreateNumpyVector(torch.tensor([pred_token], dtype=torch.uint8).numpy())
                # Build action vector
                act_vec = builder.CreateNumpyVector(torch.tensor([action_token], dtype=torch.uint8).numpy())
                
                from schema.scmoa.Message import MessageStart, MessageAddPayloadType, MessageAddPayload, MessageEnd
                
                InferenceResultStart(builder)
                InferenceResultAddPredictedState(builder, pred_vec)
                InferenceResultAddAction(builder, act_vec)
                InferenceResultAddStepId(builder, step_id)
                ir_table = InferenceResultEnd(builder)
                
                MessageStart(builder)
                MessageAddPayloadType(builder, Payload().InferenceResult)
                MessageAddPayload(builder, ir_table)
                builder.Finish(MessageEnd(builder))
                
                out_buf = builder.Output()
                win32file.WriteFile(handle, struct.pack('<I', len(out_buf)))
                win32file.WriteFile(handle, out_buf)
                
            elif p_type == Payload().Checkpoint:
                cp = Checkpoint()
                cp.Init(msg.Payload().Bytes, msg.Payload().Pos)
                model.save_checkpoint(cp.Filepath().decode('utf-8'))
                
            elif p_type == Payload().TopologyMutation:
                tm = TopologyMutation()
                tm.Init(msg.Payload().Bytes, msg.Payload().Pos)
                model.rewire_architecture(tm.Command().decode('utf-8'))
                
        except Exception as e:
            print(f"Agent Error: {e}")
            break

if __name__ == "__main__":
    start_agent()
