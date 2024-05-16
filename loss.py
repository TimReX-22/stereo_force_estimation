import torch
import torch.nn.functional as F

def gradient_x(img: torch.Tensor) -> torch.Tensor:
    return img[:, :, :, :-1] - img[:, :, :, 1:]

def gradient_y(img: torch.Tensor) -> torch.Tensor:
    return img[:, :, :-1, :] - img[:, :, 1:, :]

def SSIM(x: torch.Tensor, y: torch.Tensor, C1: float = 0.01**2, C2: float = 0.03**2) -> torch.Tensor:
    mu_x = F.avg_pool2d(x, 3, 1, padding=1)
    mu_y = F.avg_pool2d(y, 3, 1, padding=1)

    sigma_x  = F.avg_pool2d(x * x, 3, 1, padding=1) - mu_x * mu_x
    sigma_y  = F.avg_pool2d(y * y, 3, 1, padding=1) - mu_y * mu_y
    sigma_xy = F.avg_pool2d(x * y, 3, 1, padding=1) - mu_x * mu_y

    SSIM_n = (2 * mu_x * mu_y + C1) * (2 * sigma_xy + C2)
    SSIM_d = (mu_x * mu_x + mu_y * mu_y + C1) * (sigma_x + sigma_y + C2)

    SSIM = SSIM_n / SSIM_d
    return torch.clamp((1 - SSIM) / 2, 0, 1)

def image_loss(reconstructed: torch.Tensor, target: torch.Tensor, alpha: float = 0.85) -> torch.Tensor:
    l1_loss = F.l1_loss(reconstructed, target, reduction='mean')
    ssim_loss = SSIM(reconstructed, target).mean()
    return alpha * ssim_loss + (1 - alpha) * l1_loss

def disparity_smoothness_loss(disp: torch.Tensor, img: torch.Tensor) -> torch.Tensor:
    disp_gradients_x = gradient_x(disp)
    disp_gradients_y = gradient_y(disp)

    img_gradients_x = gradient_x(img)
    img_gradients_y = gradient_y(img)

    weights_x = torch.exp(-torch.mean(torch.abs(img_gradients_x), 1, keepdim=True))
    weights_y = torch.exp(-torch.mean(torch.abs(img_gradients_y), 1, keepdim=True))

    smoothness_x = disp_gradients_x * weights_x
    smoothness_y = disp_gradients_y * weights_y

    return (smoothness_x.abs().mean() + smoothness_y.abs().mean())

def total_loss(left_image: torch.Tensor, 
               right_image: torch.torch.Tensor, 
               reconstructed_right: torch.Tensor, 
               reconstructed_left: torch.Tensor, 
               disparity: torch.Tensor, 
               alpha: float = 0.85, 
               smoothness_weight: float = 0.1) -> torch.Tensor:
    right_reconstruction_loss = image_loss(reconstructed_right, right_image, alpha)
    left_reconstruction_loss = image_loss(reconstructed_left, left_image, alpha)
    
    smoothness_loss = smoothness_weight * disparity_smoothness_loss(disparity, left_image)
    reconstruction_loss = left_reconstruction_loss + right_reconstruction_loss

    return reconstruction_loss + smoothness_loss
