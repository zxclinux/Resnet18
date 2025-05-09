import torch
from ptflops import get_model_complexity_info
from models.resnet_CIFAR_18 import ResNet18Custom  # заміни на актуальний імпорт


model = ResNet18Custom(num_classes=10)


model.eval()


flops, params = get_model_complexity_info(
        model,
        input_res=(3, 32, 32),
        as_strings=True,
        print_per_layer_stat=True,
        verbose=True
    )

print(f"\n✅ Total FLOPs: {flops}\n✅ Total Params: {params}")
