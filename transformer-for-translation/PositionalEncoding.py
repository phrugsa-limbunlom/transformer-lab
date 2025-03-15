import torch
import math
import torch.nn as nn
from torch import Tensor

# max_len is the maximum length of the sequence
# PE(pos, 2i) = sin(pos / 10000^(2i/d_model)) (2i = even positions)
# PE(pos, 2i+1) = cos(pos / 10000^(2i/d_model)) (2i+1 = odd positions)
# pos is the position in the sequence
# i is the index of the embedding
# d_model is the dimension of the model (embedding size)
class PositionalEncoding(nn.Module):
    def __init__(self,
                 emb_size: int,
                 dropout: float,
                 maxlen: int = 5000):
        super(PositionalEncoding, self).__init__()

        den = torch.exp(-torch.arange(0, emb_size, 2) * math.log(10000) / emb_size) # creates a denominator for the positional encoding (set for 2i = even positions)
        pos = torch.arange(0, maxlen).reshape(maxlen, 1) # creates position indices from 0 to maxlen-1
        pos_embedding = torch.zeros((maxlen, emb_size))
        pos_embedding[:, 0::2] = torch.sin(pos * den) # fills even positions with sin values
        pos_embedding[:, 1::2] = torch.cos(pos * den) # fills odd positions with cos values
        pos_embedding = pos_embedding.unsqueeze(-2) # adds a new dimension at the second last position (Before: (maxlen, emb_size)/After: (maxlen, 1, emb_size))

        self.dropout = nn.Dropout(dropout)
        self.register_buffer('pos_embedding', pos_embedding)

    def forward(self, token_embedding: Tensor):
        return self.dropout(token_embedding + self.pos_embedding[:token_embedding.size(0), :])


