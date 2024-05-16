import os

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from torchvision import transforms

from image_reconstruction import ImageReconstructionNetwork, ImageReconstructionConfig
from dataset import StereoDataset
from loss import total_loss

from tqdm import tqdm
import matplotlib.pyplot as plt
import numpy as np

import constants

def save_and_plot_losses(num_epochs, train_losses, test_losses, dir: str = "training_plots"):
    os.makedirs(dir, exist_ok=True)
    
    plot_file = os.path.join(dir, constants.LOSS_PLOT_FN)
    loss_file = os.path.join(dir, constants.LOSSES_FN)

    epochs = np.arange(1, num_epochs + 1)
    plt.figure()
    plt.plot(epochs, train_losses, label="Train Loss")
    plt.plot(epochs, test_losses, label="Test Loss")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.legend()
    plt.savefig(plot_file)
    plt.close()

    with open(loss_file, "w") as f:
        f.write("Epoch, train_loss, test_loss\n")
        for epoch, train_loss, test_loss in zip(epochs, train_losses, test_losses):
            f.write(f"{epoch}, {train_loss:.4f}, {test_loss:.4f}\n")


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Hyperparameters
    num_layers = 18
    pretrained = False
    batch_size = 8
    num_epochs = 2
    learning_rate = 1e-4
    num_disparities = 64
    train_test_split_ratio = 0.8

    left_image_dirs = [f"data/images/dec6_force_no_TA_lastP_randomPosHeight_cs100_run{run_nr}_left" for run_nr in range(1, 4)]
    right_image_dirs = [f"data/images/dec6_force_no_TA_lastP_randomPosHeight_cs100_run{run_nr}_right" for run_nr in range(1, 4)]


    # Transformations
    transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor()
    ])

    dataset = StereoDataset(left_image_dirs, right_image_dirs, transform=transform)
    train_size = int(train_test_split_ratio * len(dataset))
    test_size = len(dataset) - train_size
    train_dataset, test_dataset = random_split(dataset, [train_size, test_size])

    dataloaders = {
        "train": DataLoader(train_dataset, batch_size=batch_size, shuffle=True),
        "test": DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    }

    config = ImageReconstructionConfig(num_layers, pretrained, num_disparities)
    model = ImageReconstructionNetwork(config)
    model.to(device)

    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.1)
    
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
                left_image, right_image = left_image.to(device), right_image.to(device)

                optimizer.zero_grad()

                with torch.set_grad_enabled(phase == "train"):
                    reconstructed_right, reconstructed_left, disparity = model(left_image, right_image)
                    loss = total_loss(left_image, right_image, reconstructed_right, reconstructed_left, disparity)
                    if phase == "train":
                        loss.backward()
                        optimizer.step()

                batch_size = left_image.size(0)
                running_loss += loss.item() * batch_size

            epoch_loss = running_loss / len(dataloaders[phase].dataset)
            print(f"{phase.capitalize()} Loss: {epoch_loss:.4f}")

            if phase == "train":
                train_losses.append(epoch_loss)
            else:
                test_losses.append(epoch_loss)

        scheduler.step()
        
    save_and_plot_losses(num_epochs, train_losses, test_losses)
    torch.save(model.state_dict(), "weights/image_reconstruction_weights.pth")



if __name__ == "__main__":
    main()
