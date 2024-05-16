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

def image_loss(reconstructed: torch.Tensor, target: torch.Tensor, mask: torch.Tensor, alpha: float = 0.85) -> torch.Tensor:
    """
    Computes the L1 and SSIM loss for the reconstructed images, but only in places where the features are visible in both images, to account for occluded features
    """
    l1_loss = F.l1_loss(reconstructed * mask, target * mask, reduction='mean')
    ssim_loss = (SSIM(reconstructed, target) * mask).mean()
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

# TODO: Double check if it makes sense to mask occluded features
def create_occlusion_mask(disparity: torch.Tensor, direction: str) -> torch.Tensor:
    batch_size, _, height, width = disparity.shape
    mask = torch.ones_like(disparity, dtype=torch.float32)

    grid_x, grid_y = torch.meshgrid(torch.arange(width, device=disparity.device), torch.arange(height, device=disparity.device))
    grid_x = grid_x.unsqueeze(0).expand(batch_size, -1, -1).float()
    
    if direction == 'right':
        occlusion = grid_x + disparity.squeeze(1) >= width
    elif direction == 'left':
        occlusion = grid_x - disparity.squeeze(1) < 0
    else:
        raise ValueError(f"Invalid direction {direction}. Use 'right' or 'left'.")

    occlusion = occlusion.unsqueeze(1)
    mask[occlusion] = 0
    
    return mask

def total_loss(left_image: torch.Tensor, 
               right_image: torch.torch.Tensor, 
               reconstructed_right: torch.Tensor, 
               reconstructed_left: torch.Tensor, 
               disparity: torch.Tensor, 
               alpha: float = 0.85, 
               smoothness_weight: float = 0.1) -> torch.Tensor:
    mask_right = create_occlusion_mask(disparity, 'right')
    mask_left = create_occlusion_mask(disparity, 'left')

    right_reconstruction_loss = image_loss(reconstructed_right, right_image, mask_right, alpha)
    left_reconstruction_loss = image_loss(reconstructed_left, left_image, mask_right, alpha)
    
    smoothness_loss = smoothness_weight * disparity_smoothness_loss(disparity, left_image)
    reconstruction_loss = left_reconstruction_loss + right_reconstruction_loss

    return reconstruction_loss + smoothness_loss
