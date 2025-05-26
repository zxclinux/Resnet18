import torch
from torchvision import transforms
from torchvision.transforms.functional import to_pil_image
from PIL import Image
import sys
import matplotlib.pyplot as plt
import torch.nn.functional as F
from models.resnetCIFAR_custom import ResNet18Custom
from torchcam.methods import GradCAM
from torchcam.utils import overlay_mask

# === Константи ===
CIFAR10_CLASSES = ['airplane', 'automobile', 'bird', 'cat', 'deer',
                   'dog', 'frog', 'horse', 'ship', 'truck']

# === Обробка аргументів ===
if len(sys.argv) != 2:
    print("⚠️  Використання: python predict.py path_to_image.jpg")
    sys.exit(1)

image_path = sys.argv[1]

# === Завантаження моделі ===
model = ResNet18Custom(num_classes=10)
model.load_state_dict(torch.load("best_model.pth", map_location="cpu"))
model.eval()

# === CAM extractor для шару 'layer4'
cam_extractor = GradCAM(model.model, target_layer='layer4')

# === Трансформація зображення
transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

# === Завантаження зображення
try:
    image = Image.open(image_path).convert("RGB")
except Exception as e:
    print(f"❌ Помилка відкриття зображення: {e}")
    sys.exit(1)

input_tensor = transform(image).unsqueeze(0)  # [1, 3, 224, 224]

# === Передбачення
output = model(input_tensor)
class_idx = output.argmax().item()

# === Grad-CAM ===
activation_map = cam_extractor(class_idx, output)

# Витягуємо активаційну карту та корегуємо розмірності
cam = activation_map[0].detach()

# Якщо активаційна карта має форму [1, 7, 7] - видаляємо перший розмір
if cam.dim() == 3:
    cam = cam.squeeze(0)  # тепер [7, 7]

# Нормалізація
cam = (cam - cam.min()) / (cam.max() - cam.min())

# Додаємо розмірності batch (N) та channel (C)
cam = cam.unsqueeze(0).unsqueeze(0)  # тепер [1, 1, 7, 7]

# Інтерполяція до розмірів оригінального зображення
cam = F.interpolate(
    cam,
    size=(image.size[1], image.size[0]),  # (висота, ширина)
    mode="bilinear",
    align_corners=False
)  # результат [1, 1, H, W]

# Видаляємо зайві розмірності
cam = cam.squeeze().cpu().numpy()  # тепер [H, W]

# Перетворюємо теплову карту в PIL Image
heatmap = to_pil_image(cam, mode="F")

# Накладаємо маску
result = overlay_mask(image, heatmap, alpha=0.5)


# === Виведення результатів
print(f"✅ Розпізнано: клас {class_idx} — {CIFAR10_CLASSES[class_idx]}")

# === Збереження результату
result.save("cam_output.png")
print("💾 Зображення з Grad-CAM збережено як cam_output.png")

# === Показ
plt.imshow(result)
plt.title(f"Class: {CIFAR10_CLASSES[class_idx]}")
plt.axis("off")
plt.show()