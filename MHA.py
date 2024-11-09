import torch
import torch.nn as nn
from SDPA import ScaledDotProductAttention

class SimpleMultiHeadAttention(nn.Module):
    def __init__(self , n_heads):
        super().__init__()
        
        self.n_heads = n_heads
        self.m = nn.ModuleList([ScaledDotProductAttention(256 , 10 , 100) for _ in range(self.n_heads)])
    
    def forward(self , x):

        out = [layer(x) for layer in self.m]
        return torch.concat(out , dim = -1)
    
class MultiHeadAttention(nn.Module):
    def __init__(self , n_heads , d_model):
        super().__init__()
        
        assert d_model % n_heads == 0 , "d_model should be perfectly divisible by n_heads"
        
        self.n_heads = n_heads
        self.d_model = d_model
        
        self.d_k = self.d_model // self.n_heads
        
        self.w_q = nn.Linear(self.d_model , self.d_model , bias = False)
        self.w_k = nn.Linear(self.d_model , self.d_model , bias = False)
        self.w_v = nn.Linear(self.d_model , self.d_model , bias = False)
        self.w_o = nn.Linear(self.d_model , self.d_model , bias = False)

    def forward(self , Q , K , V):
        
        b , n = Q.size(0) , Q.size(1)
        
        mask = torch.triu(torch.ones(n , n) , diagonal = 1).unsqueeze(0).unsqueeze(0)
        
        Q = self.w_q(Q)
        K = self.w_k(K)
        V = self.w_v(V)
        
        Q = Q.view(b , n , self.n_heads , self.d_k).transpose(1 , 2)
        K = K.view(b , n , self.n_heads , self.d_k).transpose(1 , 2)
        V = V.view(b , n , self.n_heads , self.d_k).transpose(1 , 2)
    
        attention_scores = Q @ K.transpose(-1 , -2) / (self.d_k) ** 0.5 
        
        attention_scores = attention_scores.masked_fill(mask.bool() , -torch.inf)
        
        attention_weights = torch.softmax(attention_scores , dim = -1)
        
        attention_out = attention_weights @ V

        attention_out = attention_out.transpose(2, 1).contiguous().view(b , n , self.d_model)

        out = self.w_o(attention_out)
        
        return out