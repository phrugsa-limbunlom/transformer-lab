import torch

# command to install pytorch with cuda
# pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
print("CUDA available:", torch.cuda.is_available())
print("Current device:", torch.cuda.current_device())
print("Device name:", torch.cuda.get_device_name(0)) 