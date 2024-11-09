import torch
import torch.nn as nn

class CausalMaskedAttention(nn.Module):
    def __init__(self , d_model):
        super().__init__()
        
        self.d_model = d_model
       
        self.w_q = nn.Linear(self.d_model , self.d_model , bias = False)
        self.w_k = nn.Linear(self.d_model , self.d_model , bias = False)
        self.w_v = nn.Linear(self.d_model , self.d_model , bias = False)
        self.w_o = nn.Linear(self.d_model , self.d_model , bias = False)
        
    def forward(self , x):

        n = Q.size(0)
        mask = torch.triu(torch.ones(n , n) , diagonal= 1)
        
        Q = self.w_q(x)
        K = self.w_k(x)
        V = self.w_v(x)
        
        attention_scores = Q @ K.T / (self.d_model) ** 0.5
        
        causal_attention_scores = attention_scores.masked_fill(mask.bool() , -torch.inf)
         
        attention_weights = torch.softmax(causal_attention_scores , dim = 0)
        
        attention_out = attention_weights  @ V
        
        out = self.w_o(attention_out)
        
        return out



