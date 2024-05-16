import torch
import torch.nn as nn
from disparity_estimation import DisparityEstimationNetwork, DisparityToImage

from dataclasses import dataclass
from typing import Tuple

@dataclass
class ImageReconstructionConfig:
    num_encoder_layers: int
    pretrained_encoder: bool
    
    num_disparities: int = 64

class ImageReconstructionNetwork(nn.Module):
    def __init__(self, config: ImageReconstructionConfig) -> None:
        super().__init__()
        self.disparity_model = DisparityEstimationNetwork(config.num_encoder_layers, config.pretrained_encoder, config.num_disparities)
        self.disp_to_image = DisparityToImage()
        self._initialize_weights()
        
    def forward(self, left_image: torch.Tensor, right_image: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        disparity = self.disparity_model(left_image, right_image)
        reconstructed_right = self.disp_to_image(left_image, disparity, direction="right")
        reconstructed_left = self.disp_to_image(right_image, disparity, direction="left")
        return reconstructed_right, reconstructed_left, disparity
    
    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
                nn.init.kaiming_normal_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)