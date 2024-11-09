import torch
import torch.nn as nn

class MultiQueryAttention(nn.Module):
    def __init__(self , d_model , n_heads):
        super().__init__()
        
        assert d_model % n_heads == 0 , "d_model must be divisible by n_heads"
        
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_k = self.d_model // self.n_heads
        
        self.w_q = nn.Linear(self.d_model , self.d_model , bias = False)
        self.w_k = nn.Linear(self.d_model , self.d_k , bias = False)
        self.w_v = nn.Linear(self.d_model , self.d_k , bias = False)
        self.w_o = nn.Linear(self.d_model , self.d_model , bias = False)
        
    def forward(self , Q , K , V ):
        
        b , n = Q.size(0) , Q.size(1)

        Q = self.w_q(Q)
        K = self.w_k(K)
        V = self.w_v(V)
        
        Q = Q.view(b , n , self.n_heads , self.d_k).transpose(1 , 2)
        K = K.view(b , n , 1 , self.d_k).transpose(1 , 2)
        V = V.view(b , n , 1 , self.d_k).transpose(1 , 2)

        mask = torch.triu(torch.ones(n ,  n))
        
        attention_scores = Q @ K.transpose(-2 , -1) / (self.d_k ** 0.5)
        
        attention_scores = attention_scores.masked_fill(mask.bool() , -torch.inf)
        
        attention_weights = torch.softmax(attention_scores , dim = -1)
        
        attention_out = attention_weights @ V
        
        attention_out = attention_out.transpose(1 , 2).contiguous().view(b , n , self.d_model)
        
        out = self.w_o(attention_out)
        
        return out