import torch
import torch.nn as nn
import math

def calc_rope_theta(base, dim, max_seq_len):
    freqs = torch.arange(0, dim, 2) / dim
    inv_freq = 1.0 / (base ** freqs)
    
    pos = torch.arange(max_seq_len)
    pos_emb = torch.outer(pos, inv_freq)
    return torch.polar(torch.ones_like(pos_emb), pos_emb)

def apply_rope(x, rope_theta):
    batch_size, seq_len, dim = x.shape
    
    x_complex = torch.view_as_complex(x.reshape(*x.shape[:-1], -1, 2))
    x_rotated = x_complex * rope_theta[:seq_len, :(dim//2)]
    return torch.view_as_real(x_rotated).reshape(*x.shape)

class RoPEAttention(nn.Module):
    def __init__(self, d_model, n_heads):
        super().__init__()
        assert d_model % n_heads == 0, "d_model must be divisible by n_heads"
        
        self.n_heads = n_heads
        self.d_model = d_model
        self.d_k = d_model // n_heads  
        
        self.w_q = nn.Linear(d_model, d_model)
        self.w_k = nn.Linear(d_model, d_model)
        self.w_v = nn.Linear(d_model, d_model)
        self.w_o = nn.Linear(d_model, d_model)
        
        self.register_buffer(
            "rope_theta",
            calc_rope_theta(10000, self.d_k, 1024)
        )
        
    def forward(self, q, k, v, mask=None):
        batch_size, seq_len, _ = q.shape

        q = self.w_q(q) 
        k = self.w_k(k)
        v = self.w_v(v)
        
        q = q.view(batch_size, seq_len, self.n_heads, self.d_k)
        k = k.view(batch_size, seq_len, self.n_heads, self.d_k)
        v = v.view(batch_size, seq_len, self.n_heads, self.d_k)
        
        q = apply_rope(q.reshape(-1, seq_len, self.d_k), self.rope_theta)
        k = apply_rope(k.reshape(-1, seq_len, self.d_k), self.rope_theta)
        
        q = q.view(batch_size, seq_len, self.n_heads, self.d_k)
        k = k.view(batch_size, seq_len, self.n_heads, self.d_k)
        
        q = q.transpose(1, 2) 
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)
        
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.d_k)
        
        if mask is not None:
            scores = scores.masked_fill(mask == 0, float('-inf'))

        attn = torch.softmax(scores, dim=-1)
        out = torch.matmul(attn, v)  
        
        out = out.transpose(1, 2).contiguous() 
        out = out.view(batch_size, seq_len, self.d_model)  

        return self.w_o(out)
    
batch_size = 1
seq_length = 32
d_model = 512    
n_heads = 8      

x = torch.randn(batch_size, seq_length, d_model)

attention = RoPEAttention(d_model=d_model, n_heads=n_heads)

output = attention(x, x, x)
print(f"Output shape: {output.shape}") 
