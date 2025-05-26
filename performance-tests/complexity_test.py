import torch
from torchvision.models import resnet18
from ptflops import get_model_complexity_info

model = resnet18()
with torch.cuda.device(0):
    macs, params = get_model_complexity_info(model, (3, 224, 224), as_strings=True,
                                              print_per_layer_stat=True, verbose=True)
print(f"FLOPs: {macs}, Params: {params}")
