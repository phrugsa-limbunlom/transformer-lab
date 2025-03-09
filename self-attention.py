import torch
import torch.nn as nn
import torch.nn.functional as F

class SelfAttention(nn.Module):
    def __init__(self, d_model=2,row_dim=0, col_dim=1):
        super(SelfAttention, self).__init__()

        self.W_q = nn.Linear(in_features=d_model, out_features=d_model, bias=False)
        self.W_k = nn.Linear(in_features=d_model, out_features=d_model, bias=False)
        self.W_v = nn.Linear(in_features=d_model, out_features=d_model, bias=False)

        self.row_dim = row_dim
        self.col_dim = col_dim

    def forward(self, token_encodings):
        q = self.W_q(token_encodings)
        k = self.W_k(token_encodings)
        v = self.W_v(token_encodings)

        sims = torch.matmul(q, k.transpose(dim0=self.row_dim, dim1=self.col_dim))

        scaled_sims = sims / torch.tensor(k.size(self.col_dim)**0.5) #root x = x^0.5

        attention_percents = F.softmax(scaled_sims, dim=self.col_dim)

        attention_scores = torch.matmul(attention_percents, v)

        return attention_scores

encodings_matrix = torch.tensor([[1.16, 0.23],
                                 [0.57, 1.36],
                                 [4.41, -2.16]]) #row = word, column = embedding dimension

torch.manual_seed(42) #ensure any random operations will produce the same results

selfAttention = SelfAttention(d_model=2, row_dim=0, col_dim=1)

print("Calculate self attention and display final output: ")
print(selfAttention(encodings_matrix))

## print out the weight matrix that creates the queries, keys, values
print("Calculate self attention and display step by step calculation: ")
print(selfAttention.W_q.weight.transpose(0,1))

print(selfAttention.W_k.weight.transpose(0,1))

print(selfAttention.W_v.weight.transpose(0,1))

## calculate the queries, keys, and values
q = selfAttention.W_q(encodings_matrix)
k = selfAttention.W_k(encodings_matrix)
v = selfAttention.W_v(encodings_matrix)

sims = torch.matmul(q, k.transpose(dim0=0, dim1=1))
print(sims)

scaled_sims = sims / (torch.tensor(2)**0.5)
print(scaled_sims)

attention_percents = F.softmax(scaled_sims, dim=1)
print(attention_percents)

print("Final output: ")
print(torch.matmul(attention_percents, selfAttention.W_v(encodings_matrix)))




