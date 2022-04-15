import math
import torch
from torch import nn, Tensor

class PositionalEncoding(nn.Module):
    
    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 5000):
        super().__init__()

        self.d_model = d_model #maximum length of the sequence
        self.max_len = max_len #maximum length of the sequence
        self.dropout = nn.Dropout(p=dropout)

        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len,1, d_model)
        pe[:,0, 0::2] = torch.sin(position * div_term)
        pe[:,0, 1::2] = torch.cos(position * div_term)
        
        self.register_buffer('pe', pe)

    def forward(self, x: Tensor, steps, len) -> Tensor:
        """
        Args:
            x: Tensor, shape [batch_size, seq_len, embedding_dim]
            steps: Normalized steps in the sequence. # batch(512),seq length (20)
            len: length of the data in the data.
            self.pe = 5000,75
        """
        #x = torch.transpose(x,0,1) # now, seq_len is the first location
        
        NORM = False
        batch, seq_len,dim = x.shape
        x = x * math.sqrt(self.d_model)
        if NORM:
            len = torch.unsqueeze(len,1)
            steps = steps/(len+2)
            recv_steps = steps * self.max_len
            recv_steps = torch.round(recv_steps)
            recv_steps = torch.minimum(recv_steps, torch.tensor(self.max_len-1).cuda())
            emb_steps = self.pe[recv_steps.type(torch.LongTensor),:,:x.size(2)]
        else:
            emb_steps = self.pe[steps.type(torch.LongTensor),:,:x.size(2)]
        emb_steps = emb_steps[:,:,0,:]
        #print("emb_steps",emb_steps.shape)
        x = x + emb_steps

        #before
        #x = x + self.pe[:x.size(0),:,:x.size(2)]
        
        #x = torch.transpose(x,0,1) 
        #return self.dropout(x)
        return x