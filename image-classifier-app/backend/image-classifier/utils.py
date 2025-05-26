import torch
from torchvision import transforms
from torchvision.transforms.functional import to_pil_image
from PIL import Image
import torch.nn.functional as F
from models.resnetCIFAR_custom import ResNet18Custom
from torchcam.methods import GradCAM
from torchcam.utils import overlay_mask

CIFAR10_CLASSES = ['airplane', 'automobile', 'bird', 'cat', 'deer',
                   'dog', 'frog', 'horse', 'ship', 'truck']

def load_model(model_path='models/best_model.pth'):

    model = ResNet18Custom(num_classes=10)
    model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
    model.eval()
    return model

def preprocess_image(image: Image.Image) -> torch.Tensor:

    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    return transform(image).unsqueeze(0)

def predict(image: Image.Image, model) -> dict:

    input_tensor = preprocess_image(image)
    with torch.no_grad():
        output = model(input_tensor)
        predicted_idx = torch.argmax(output, dim=1).item()
    return {
        'class': CIFAR10_CLASSES[predicted_idx],
        'class_idx': predicted_idx,
        'probabilities': torch.softmax(output, dim=1).squeeze().tolist()
    }


def generate_cam(image: Image.Image, model: torch.nn.Module, class_idx: int) -> Image.Image:

    model.load_state_dict(torch.load("models/best_model.pth", map_location="cpu"))
    model.eval()

    cam_extractor = GradCAM(model.model, target_layer='layer4')

    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])

    input_tensor = transform(image).unsqueeze(0)  # [1, 3, 224, 224]

    output = model(input_tensor)
    print(torch.softmax(output, dim=1).squeeze().tolist())
    class_idx_new = output.argmax().item()

    print(CIFAR10_CLASSES[class_idx_new])

    activation_map = cam_extractor(class_idx_new, output)

    cam = activation_map[0].detach()

    if cam.dim() == 3:
        cam = cam.squeeze(0)  # тепер [7, 7]

    cam = (cam - cam.min()) / (cam.max() - cam.min())

    cam = cam.unsqueeze(0).unsqueeze(0)  #

    cam = F.interpolate(
        cam,
        size=(image.size[1], image.size[0]),
        mode="bilinear",
        align_corners=False
    )  # результат [1, 1, H, W]

    cam = cam.squeeze().cpu().numpy()  # тепер [H, W]

    heatmap = to_pil_image(cam, mode="F")

    result = overlay_mask(image, heatmap, alpha=0.5)

    for handle in cam_extractor.hook_handles:
        handle.remove()
    cam_extractor.hook_handles.clear()

    return result