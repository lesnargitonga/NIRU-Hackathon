import sys

try:
    import torch
except Exception as e:
    print("ERROR: Failed to import torch:", e)
    sys.exit(1)

print("cuda_available=", torch.cuda.is_available())
print("torch_version=", torch.__version__)
print("cuda_version=", torch.version.cuda)
print("device_count=", torch.cuda.device_count())
if torch.cuda.is_available() and torch.cuda.device_count() > 0:
    print("device_name0=", torch.cuda.get_device_name(0))
else:
    print("device_name0=", None)
