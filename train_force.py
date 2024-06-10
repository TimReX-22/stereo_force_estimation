import os
from typing import List
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from torchvision import transforms
from image_reconstruction import DisparityEstimationConfig
from force_estimation import InteractionForcePredictionNetwork
from dataset import StereoForceDataset
from loss import total_loss
from tqdm import tqdm
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import constants
import util


def force_loss(predicted_force, true_force):
    return nn.MSELoss()(predicted_force, true_force)


def combined_loss(left_image, right_image, reconstructed_right, reconstructed_left, disparity, predicted_force, true_force):
    reconstruction_loss = total_loss(
        left_image, right_image, reconstructed_right, reconstructed_left, disparity)
    force_pred_loss = force_loss(predicted_force, true_force)
    return reconstruction_loss + force_pred_loss


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
    force_files = util.get_force_files(runs=range(1, 4))

    transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor()
    ])

    dataset = StereoForceDataset(force_files, transform=transform)
    train_size = int(train_test_split_ratio * len(dataset))
    test_size = len(dataset) - train_size
    train_dataset, test_dataset = random_split(
        dataset, [train_size, test_size])
    dataloaders = {
        "train": DataLoader(train_dataset, batch_size=batch_size, shuffle=True),
        "test": DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    }
    print(
        f"[INFO] Loaded Trainset with {train_size} samples and Testset with {test_size} samples!")
    disparity_config = DisparityEstimationConfig(
        num_layers, pretrained, num_disparities)
    model = InteractionForcePredictionNetwork(disparity_config)
    model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.1)

    train_losses = []
    test_losses = []
    train_mse = []
    test_mse = []
    train_disparity_loss = []
    test_disparity_loss = []

    for epoch in range(num_epochs):
        print(f"Epoch [{epoch + 1}/{num_epochs}]")
        for phase in ["train", "test"]:
            if phase == "train":
                model.train()
            else:
                model.eval()
            running_loss = 0.0
            running_mse = 0.0
            running_disparity_loss = 0.0
            for left_image, right_image, true_force in tqdm(dataloaders[phase]):
                left_image, right_image, true_force = left_image.to(
                    device), right_image.to(device), true_force.to(device)
                optimizer.zero_grad()
                with torch.set_grad_enabled(phase == "train"):
                    predicted_force, reconstructed_right, reconstructed_left, disparity = model(
                        left_image, right_image)
                    loss, disparity_loss, mse_loss = combined_loss(
                        left_image, right_image, reconstructed_right, reconstructed_left, disparity, predicted_force, true_force)
                    if phase == "train":
                        loss.backward()
                        optimizer.step()
                batch_size = left_image.size(0)
                running_loss += loss.item() * batch_size
                running_mse += mse_loss.item() * batch_size
                running_disparity_loss += disparity_loss.item() * batch_size
            epoch_loss = running_loss / len(dataloaders[phase].dataset)
            epoch_mse = running_mse / len(dataloaders[phase].dataset)
            epoch_disparity_loss = running_disparity_loss / \
                len(dataloaders[phase].dataset)
            if phase == "train":
                train_losses.append(epoch_loss)
                train_mse.append(epoch_mse)
                train_disparity_loss.append(epoch_disparity_loss)
            else:
                test_losses.append(epoch_loss)
                test_mse.append(epoch_mse)
                test_disparity_loss.append(epoch_disparity_loss)
        print(f"Train Loss: {train_losses[-1]:.4f} \t Test Loss: {test_losses[-1]:.4f} \t Train MSE: {train_mse[-1]:.4f} \t Test MSE: {test_mse[-1]:.4f} \t Train Disparity Loss: {train_disparity_loss[-1]:.4f} \t Test Disparity Loss: {test_disparity_loss[-1]:.4f}")
        scheduler.step()

    util.save_and_plot_losses(num_epochs, train_losses,
                              test_losses, "Combined Loss", "Loss")
    util.save_and_plot_losses(num_epochs, train_mse,
                              test_mse, "Force MSE", "MSE")
    util.save_and_plot_losses(num_epochs, train_disparity_loss,
                              test_disparity_loss, "Disparity Loss", "Loss")
    torch.save(model.state_dict(), "weights/force_estimation_weights.pth")


if __name__ == "__main__":
    main()
