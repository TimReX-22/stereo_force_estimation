import torch
import torch.nn as nn
import torch.nn.functional as F
from monodepth2.networks.resnet_encoder import ResnetEncoder
from monodepth2.networks.depth_decoder import ModifiedDepthDecoder
from simnet.lib.net.models.simplenet import DotProductCostVolume, SoftArgmin


class DisparityEstimationNetwork(nn.Module):
    def __init__(self, num_layers: int, pretrained: bool, num_disparities: int = 64) -> None:
        super().__init__()
        self.encoder = ResnetEncoder(num_layers, pretrained)
        self.decoder = ModifiedDepthDecoder(num_ch_enc=self.encoder.num_ch_enc)
        self.cost_volume = DotProductCostVolume(num_disparities)
        self.soft_argmin = SoftArgmin()

    def forward(self, left_image: torch.Tensor, right_image: torch.Tensor) -> torch.Tensor:
        left_features = self.encoder(left_image)
        right_features = self.encoder(right_image)

        left_decoded = self.decoder(left_features)
        right_decoded = self.decoder(right_features)

        cost_volume = self.cost_volume(
            left_decoded[('output', 0)], right_decoded[('output', 0)])

        disparity = self.soft_argmin(cost_volume)

        return disparity


class DisparityToImage(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, image: torch.Tensor, disparity_map: torch.Tensor, direction: str):
        batch_size, _, height, width = image.shape

        x_base = torch.linspace(-1, 1, width, device=image.device).repeat(
            batch_size, height, 1)
        y_base = torch.linspace(-1, 1, height, device=image.device).repeat(
            batch_size, width, 1).transpose(1, 2)

        # normalize disparity map, same range as grid
        x_shifts = disparity_map.squeeze(1) / (width / 2 )

        if direction == 'right':
            x_new = x_base + x_shifts
        elif direction == 'left':
            x_new = x_base - x_shifts
        else:
            raise ValueError(
                f"Invalid direction {direction}. Use 'right' or 'left'.")

        assert x_new.shape == y_base.shape, f"{x_new.shape=}, {y_base.shape=}"
        grid = torch.stack((x_new, y_base), dim=-1).squeeze(1)

        warped_image = F.grid_sample(
            image, grid, mode='bilinear', padding_mode='zeros', align_corners=True)

        return warped_image
