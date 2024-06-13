import os
from typing import List

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from torchvision import transforms

from image_reconstruction import ImageReconstructionNetwork, DisparityEstimationConfig
from dataset import StereoDataset
from loss import total_loss

from tqdm import tqdm
import matplotlib.pyplot as plt
import numpy as np

import util


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Hyperparameters
    num_layers = 18
    pretrained = True
    batch_size = 8
    num_epochs = 20
    learning_rate = 1e-5
    num_disparities = 64
    train_test_split_ratio = 0.8

    left_image_dirs = util.get_image_dirs(cam="left", runs=range(1, 4))
    right_image_dirs = util.get_image_dirs(cam="right", runs=range(1, 4))

    transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor()
    ])

    dataset = StereoDataset(
        left_image_dirs, right_image_dirs, transform=transform)
    train_size = int(train_test_split_ratio * len(dataset))
    test_size = len(dataset) - train_size
    train_dataset, test_dataset = random_split(
        dataset, [train_size, test_size])

    dataloaders = {
        "train": DataLoader(train_dataset, batch_size=batch_size, shuffle=True),
        "test": DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    }

    config = DisparityEstimationConfig(num_layers, pretrained, num_disparities)
    model = ImageReconstructionNetwork(config)
    model.to(device)

    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.1)

    train_losses = []
    test_losses = []

    for epoch in range(num_epochs):
        print(f"Epoch [{epoch + 1}/{num_epochs}]")
        for phase in ["train", "test"]:
            if phase == "train":
                model.train()
            else:
                model.eval()

            running_loss = 0.0
            for left_image, right_image in tqdm(dataloaders[phase]):
                left_image, right_image = left_image.to(
                    device), right_image.to(device)

                optimizer.zero_grad()

                with torch.set_grad_enabled(phase == "train"):
                    reconstructed_right, reconstructed_left, disparity = model(
                        left_image, right_image)
                    loss = total_loss(
                        left_image, right_image, reconstructed_right, reconstructed_left, disparity)
                    if phase == "train":
                        loss.backward()
                        optimizer.step()

                batch_size = left_image.size(0)
                running_loss += loss.item() * batch_size

            epoch_loss = running_loss / len(dataloaders[phase].dataset)

            if phase == "train":
                train_losses.append(epoch_loss)
            else:
                test_losses.append(epoch_loss)

        print(
            f"Train Loss: {train_losses[-1]:.4f} \t Test Loss: {test_losses[-1]:.4f}")
        scheduler.step()

    util.save_and_plot_losses(num_epochs, train_losses, test_losses)
    torch.save(model.state_dict(), "weights/image_reconstruction_weights.pth")


if __name__ == "__main__":
    main()
