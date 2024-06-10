import torch
from torchvision import transforms
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np

from image_reconstruction import ImageReconstructionNetwork, DisparityEstimationConfig
from typing import Optional


def load_image(image_path: str, transform: Optional[transforms.Compose] = None):
    image = Image.open(image_path).convert('RGB')
    if transform:
        image = transform(image).unsqueeze(0)
    return image


def save_image(img_tensor: torch.Tensor, path: str):
    image = img_tensor.squeeze(0).permute(1, 2, 0).cpu().numpy()
    image = (image * 255).astype(np.uint8)
    image = Image.fromarray(image)
    image.save(path)


def save_disparity(disparity: torch.Tensor, path: str):
    disparity = disparity.squeeze(0).squeeze(0).cpu().numpy()
    disparity = (disparity / np.max(disparity) * 255).astype(np.uint8)
    disparity_image = Image.fromarray(disparity)
    disparity_image.save(path)


def visualize_disparity(disparity: torch.Tensor):
    plt.imshow(disparity.squeeze(0).squeeze(0).cpu().numpy(), cmap='gray')
    plt.colorbar()
    plt.show()


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    left_image_path = "data/images/dec6_force_no_TA_lastP_randomPosHeight_cs100_run1_left/zed_left_1.png"
    right_image_path = "data/images/dec6_force_no_TA_lastP_randomPosHeight_cs100_run1_right/zed_right_1.png"
    weights_path = "weights/image_reconstruction_weights.pth"

    transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor()
    ])

    left_image = load_image(left_image_path, transform).to(device)
    right_image = load_image(right_image_path, transform).to(device)

    config = DisparityEstimationConfig(
        num_encoder_layers=34, pretrained_encoder=False, num_disparities=64)
    model = ImageReconstructionNetwork(config).to(device)
    model.load_state_dict(torch.load(weights_path))

    model.eval()
    with torch.no_grad():
        reconstructed_right, _, disparity = model(left_image, right_image)

    print(f"{left_image.shape=}")
    print(f"{reconstructed_right.shape=}")
    print(f"{disparity.shape=}")

    visualize_disparity(disparity)
    print(f"{disparity=}")

    save_image(reconstructed_right, "evals/reconstructed_right.png")
    save_disparity(disparity, "evals/disparity.png")


if __name__ == "__main__":
    main()
