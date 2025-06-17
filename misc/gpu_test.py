# Quick python script to check pytorch can find the GPU
# Should return CUDA = True and Num GPUs = 1
import torch

print("CUDA available?:", torch.cuda.is_available())
print("Number of GPUs:", torch.cuda.device_count())

if torch.cuda.is_available():
    print("GPU name:", torch.cuda.get_device_name(0))
