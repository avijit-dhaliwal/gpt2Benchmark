import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.cpp_extension import load
import numpy as np
import jax
import jax.numpy as jnp
from typing import List, Tuple, Optional
import ray
from fastapi import FastAPI
from pydantic import BaseModel
import asyncio
import onnxruntime as ort
from transformers import GPT2Tokenizer
import os

def custom_matmul(A, B):
    return np.matmul(A, B)

class GPT2Advanced(nn.Module):
    def __init__(self, vocab_size: int, d_model: int, n_heads: int, n_layers: int, max_seq_len: int):
        super().__init__()
        self.vocab_size = vocab_size
        self.d_model = d_model
        self.n_heads = n_heads
        self.n_layers = n_layers
        self.max_seq_len = max_seq_len

        self.token_embedding = nn.Embedding(vocab_size, d_model)
        self.position_embedding = nn.Embedding(max_seq_len, d_model)
        self.layers = nn.ModuleList([TransformerBlock(d_model, n_heads) for _ in range(n_layers)])
        self.ln_f = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, vocab_size, bias=False)

        self.tokenizer = GPT2Tokenizer.from_pretrained('gpt2')

        # JAX-based function for fast CPU computations
        self.jax_forward = jax.jit(self.jax_forward_impl)

        # Initialize ONNX session only if the file exists
        onnx_path = "models/gpt2_model.onnx"
        if os.path.exists(onnx_path):
            self.onnx_session = ort.InferenceSession(onnx_path)
        else:
            self.onnx_session = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, t = x.size()
        assert t <= self.max_seq_len, f"Cannot forward sequence of length {t}, max is {self.max_seq_len}"
        pos = torch.arange(0, t, dtype=torch.long, device=x.device).unsqueeze(0)

        tok_emb = self.token_embedding(x)
        pos_emb = self.position_embedding(pos)
        x = tok_emb + pos_emb

        for layer in self.layers:
            x = layer(x)

        x = self.ln_f(x)
        logits = self.head(x)

        return logits

    @staticmethod
    def jax_forward_impl(x: jnp.ndarray) -> jnp.ndarray:
        # Simplified JAX implementation for demonstration
        return jax.nn.softmax(x)

    def generate(self, prompt: str, max_length: int = 100) -> str:
        input_ids = self.tokenizer.encode(prompt, return_tensors="pt")
        attention_mask = torch.ones(input_ids.shape, dtype=torch.long)
        
        for _ in range(max_length):
            with torch.no_grad():
                outputs = self(input_ids)
            next_token_logits = outputs[:, -1, :]
            next_token = torch.argmax(next_token_logits, dim=-1).unsqueeze(-1)
            input_ids = torch.cat([input_ids, next_token], dim=-1)
            attention_mask = torch.cat([attention_mask, torch.ones((1, 1), dtype=torch.long)], dim=-1)
            
            if next_token.item() == self.tokenizer.eos_token_id:
                break
        
        return self.tokenizer.decode(input_ids[0])

    def parallel_generate(self, prompts: List[str]) -> List[str]:
        return [self.generate(prompt) for prompt in prompts]

    def onnx_inference(self, input_ids: np.ndarray) -> np.ndarray:
        if self.onnx_session is None:
            raise ValueError("ONNX session is not initialized. Make sure the ONNX model exists.")
        ort_inputs = {self.onnx_session.get_inputs()[0].name: input_ids}
        ort_outputs = self.onnx_session.run(None, ort_inputs)
        return ort_outputs[0]

    def numpy_matmul(self, A: np.ndarray, B: np.ndarray) -> np.ndarray:
        return custom_matmul(A, B)

    

class TransformerBlock(nn.Module):
    def __init__(self, d_model: int, n_heads: int):
        super().__init__()
        self.attn = nn.MultiheadAttention(d_model, n_heads)
        self.ff = nn.Sequential(
            nn.Linear(d_model, 4 * d_model),
            nn.GELU(),
            nn.Linear(4 * d_model, d_model)
        )
        self.ln1 = nn.LayerNorm(d_model)
        self.ln2 = nn.LayerNorm(d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.ln1(x), self.ln1(x), self.ln1(x))[0]
        x = x + self.ff(self.ln2(x))
        return x

# FastAPI for serving the model
app = FastAPI()

class GenerationRequest(BaseModel):
    prompt: str
    max_length: int = 100

@app.post("/generate")
async def generate_text(request: GenerationRequest):
    model = GPT2Advanced(50257, 768, 12, 12, 1024)  # Load or initialize your model
    generated_text = model.generate(request.prompt, request.max_length)
    return {"generated_text": generated_text}

if __name__ == "__main__":
    model = GPT2Advanced(50257, 768, 12, 12, 1024)
    
    # Demonstrate various functionalities
    print(model.generate("Once upon a time"))
    
    prompts = ["Hello world", "AI is amazing"]
    results = model.parallel_generate(prompts)
    print(results)
    
    input_ids = np.random.randint(0, 50257, (1, 10))
    onnx_output = model.onnx_inference(input_ids)
    print("ONNX output shape:", onnx_output.shape)
    
    A = np.random.rand(1024, 1024).astype(np.float32)
    B = np.random.rand(1024, 1024).astype(np.float32)
    C = model.numpy_matmul(A, B)
    print("NumPy matmul output shape:", C.shape)
    
    # Trace the entire model
    x = torch.randint(0, 50257, (1, 10))
    traced_model = torch.jit.trace(model, x)
    print("TorchScript output:", traced_model(x).shape)
    # Run FastAPI server
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)