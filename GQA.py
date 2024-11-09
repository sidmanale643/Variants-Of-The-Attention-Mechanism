import torch
import torch.nn as nn

class GroupedMultiQueryAttention(nn.Module):
    def __init__(self, d_model, q_heads, kv_heads):
        super().__init__()
        
        assert d_model % q_heads == 0, "d_model must be divisible by num query heads"
        assert d_model % kv_heads == 0, "d_model must be divisible by num kv heads"

        self.d_model = d_model
        self.q_heads = q_heads
        self.kv_heads = kv_heads
        
        self.d_k = d_model // q_heads

        self.w_q = nn.Linear(self.d_model ,  self.q_heads  *  self.d_k , bias = False)
        self.w_k = nn.Linear(self.d_model ,  self.kv_heads *  self.d_k , bias = False)
        self.w_v = nn.Linear(self.d_model ,  self.kv_heads *  self.d_k , bias = False)
        
        self.w_o = nn.Linear(self.d_model, self.d_model)
        
    def forward(self, Q, K, V):
        b , n , _ = Q.size()
        q_per_kv_head = self.q_heads // self.kv_heads
        
        Q = self.w_q(Q)
        K = self.w_k(K)
        V = self.w_v(V)

        Q = Q.view(b, n, self.kv_heads , q_per_kv_head, self.d_k).transpose(1, 2)  
        K = K.view(b, n, self.kv_heads ,    1         , self.d_k).transpose(1, 2)  
        V = V.view(b, n, self.kv_heads ,    1         , self.d_k).transpose(1, 2) 
        
        attention_scores = Q @ K.transpose(-1 , -2) / (self.d_k ** 0.5)
        
        attention_weights = torch.softmax(attention_scores , dim = -1)
        attention_out = attention_weights @ V        
        out =  attention_out.transpose(1,2).contiguous().view(b , n , -1)
        return out