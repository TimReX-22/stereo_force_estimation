import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models.detection.backbone_utils import resnet_fpn_backbone
from torchvision.models.resnet import ResNet50_Weights, ResNet101_Weights
from disparity_estimation import DisparityEstimationNetwork, DisparityToImage, DisparityEstimationConfig
from image_reconstruction import ImageDirection

from typing import Tuple


class DownsampleDisparity(nn.Module):
    def __init__(self):
        super().__init__()
        self.layer1 = nn.Sequential(
            nn.Conv2d(1, 2, kernel_size=3, padding=1),
            nn.AvgPool2d(kernel_size=2, stride=2),
            nn.ReLU()
        )
        self.layer2 = nn.Sequential(
            nn.Conv2d(2, 4, kernel_size=3, padding=1),
            nn.AvgPool2d(kernel_size=2, stride=2),
            nn.ReLU()
        )
        self.conv_final = nn.Conv2d(
            4, 4, kernel_size=3, padding=1)
        self.relu_final = nn.ReLU()

    def forward(self, disparity: torch.Tensor) -> torch.Tensor:
        x = self.layer1(disparity)
        x = self.layer2(x)
        # x = self.relu_final(self.conv_final(x))
        return x


class ForcePredictionNetwork(nn.Module):
    def __init__(self, disparity_config: DisparityEstimationConfig) -> None:
        super().__init__()
        self.disparity_model = DisparityEstimationNetwork(disparity_config)
        self.downsample_disparity = DownsampleDisparity()
        self.resnet_fpn = resnet_fpn_backbone(
            'resnet50', weights=ResNet50_Weights.DEFAULT)
        self.resnet_fpn_large = resnet_fpn_backbone(
            'resnet101', weights=ResNet101_Weights.DEFAULT)
        self.conv_reduce_channels = nn.Conv2d(260, 3, kernel_size=1)

        self.fc = nn.Sequential(
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 3)
        )
        self.disp_to_image = DisparityToImage()

    def forward(self, left_image: torch.Tensor, right_image: torch.Tensor) -> Tuple[torch.Tensor,
                                                                                    torch.Tensor,
                                                                                    torch.Tensor,
                                                                                    torch.Tensor,
                                                                                    torch.Tensor]:
        disparity = self.disparity_model(left_image, right_image)
        downsampled_disparity = self.downsample_disparity(disparity)
        fpn_features = self.resnet_fpn(left_image)

        concatenated_features = torch.cat(
            (fpn_features['0'], downsampled_disparity), dim=1)

        reduced_channels_features = self.conv_reduce_channels(
            concatenated_features)

        fpn_large_features = self.resnet_fpn_large(reduced_channels_features)

        pooled_features = F.adaptive_avg_pool2d(
            fpn_large_features['0'], (1, 1))
        pooled_features = torch.flatten(pooled_features, 1)

        interaction_force = self.fc(pooled_features)

        reconstructed_right = self.disp_to_image(
            left_image, disparity, direction=ImageDirection.RIGHT)
        reconstructed_left = self.disp_to_image(
            right_image, disparity, direction=ImageDirection.LEFT)

        return interaction_force, reconstructed_right, reconstructed_left, disparity


if __name__ == "__main__":

    disparity_config = DisparityEstimationConfig(18, True)
    interaction_force_model = ForcePredictionNetwork(
        disparity_config)

    left_image = torch.randn((1, 3, 256, 256))
    right_image = torch.randn((1, 3, 256, 256))

    interaction_force, reconstructed_right, reconstructed_left, disparity = interaction_force_model(
        left_image, right_image)
    print(interaction_force)
    print(reconstructed_right.shape, reconstructed_left.shape, disparity.shape)
