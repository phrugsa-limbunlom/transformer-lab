import torch
import torch.nn as nn
import torch.nn.functional as F

class Attention(nn.Module):
    def __init__(self, d_model=2, row_dim=0, col_dim=1):
        super(Attention, self).__init__()

        self.W_q = nn.Linear(in_features=d_model, out_features=d_model, bias=False)
        self.W_k = nn.Linear(in_features=d_model, out_features=d_model, bias=False)
        self.W_v = nn.Linear(in_features=d_model, out_features=d_model, bias=False)

        self.row_dim = row_dim
        self.col_dim = col_dim

    def forward(self, encodings_for_q, encodings_for_k, encodings_for_v, mask=None):

        q = self.W_q(encodings_for_q)
        k = self.W_k(encodings_for_k)
        v = self.W_v(encodings_for_v)

        sims = torch.matmul(q, k.transpose(dim0=self.row_dim, dim1=self.col_dim))

        scaled_sims = sims / torch.tensor(k.size(self.col_dim)**0.5)

        if mask is not None:
            scaled_sims = sims.masked_fill(mask=mask, value=-1e9)

        attention_percents = F.softmax(scaled_sims, dim=self.col_dim)

        attention_scores = torch.matmul(attention_percents, v)

        return attention_scores

## create matrices of token encodings...
encodings_for_q = torch.tensor([[1.16, 0.23],
                                [0.57, 1.36],
                                [4.41, -2.16]])

encodings_for_k = torch.tensor([[1.16, 0.23],
                                [0.57, 1.36],
                                [4.41, -2.16]])

encodings_for_v = torch.tensor([[1.16, 0.23],
                                [0.57, 1.36],
                                [4.41, -2.16]])

## set the seed for the random number generator
torch.manual_seed(42)

## create an attention object
attention = Attention(d_model=2,
                      row_dim=0,
                      col_dim=1)

## calculate encoder-decoder attention
attention(encodings_for_q, encodings_for_k, encodings_for_v)

class MultiHeadAttention(nn.Module):
    def __init__(self, d_model=2, num_heads=2, row_dim=0, col_dim=1):

        super(MultiHeadAttention, self).__init__()

        ## create a bunch of attention heads
        self.heads = nn.ModuleList(
            [Attention(d_model, row_dim, col_dim) for _ in range(num_heads)]
        )

        self.col_dim = col_dim

    def forward(self, encodings_for_q, encodings_for_k, encodings_for_v):
        return torch.cat([head(encodings_for_q, encodings_for_k, encodings_for_v) for head in self.heads], dim=self.col_dim)

## set the seed for the random number generator
torch.manual_seed(42)

## create an attention object
multiHeadAttention = MultiHeadAttention(d_model=2,
                                        row_dim=0,
                                        col_dim=1,
                                        num_heads=2)

## calculate encoder-decoder attention
result = multiHeadAttention(encodings_for_q, encodings_for_k, encodings_for_v)

print(result)