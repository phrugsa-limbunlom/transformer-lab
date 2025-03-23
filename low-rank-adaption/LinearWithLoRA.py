import torch
from LoRALayer import LoRALayer

class LinearWithLoRA(torch.nn.Module):
    def __init__(self, linear, rank, alpha, device):
        super().__init__()
        self.linear = linear.to(device)
        self.lora = LoRALayer(
            linear.in_features, linear.out_features, rank, alpha
        ).to(device)

    def forward(self, x):
        
        return self.linear(x) + self.lora(x)