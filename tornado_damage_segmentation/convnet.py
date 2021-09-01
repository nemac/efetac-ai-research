import numpy as np
import torch
from torch.utils.data import DataLoader, random_split
import torchvision
from matplotlib import pyplot as plt
import segmentation_models_pytorch as smp
from config import raster_dir, mask_dir
from neural_net import NeuralNet
from rasterio_dataset import RasterioDataset

# Parameters
test_set_size = 50
num_training_epochs = 10
test_epoch_period = 2
model = NeuralNet(loss=torch.nn.CrossEntropyLoss())

# Load data
data = RasterioDataset(raster_dir, mask_dir, transform=torchvision.transforms.ToTensor())

# Split into training and testing sets and create data loaders
train_data, test_data = random_split(data, [len(data) - test_set_size, test_set_size])
train_loader = DataLoader(train_data, batch_size=10, shuffle=True)
test_loader = DataLoader(test_data)


# Tests the neural network, returning average accuracy
# TODO update accuracy measure. It can currently get very high by guessing all zeros
def test(verbose=False):
    accuracies = []
    for raster, mask in test_loader:
        print("raster shape: " + str(raster.shape))
        print("mask shape: " + str(mask.shape))
        output = model(raster)
        print("output shape: " + str(output.shape))
        smp.losses.JaccardLoss(mode='binary').forward()
        accuracies.append(np.average(mask == output.argmax(dim=1)))
        if verbose:
            print(accuracies[-1])
    return np.average(accuracies)


# Train the neural network
"""accuracies = []
for i in range(num_training_epochs):
    if i % test_epoch_period == 0:
        accuracies.append(test())
        print(i, accuracies[-1])
    for rasters, masks in train_loader:
        output = model(rasters)
        # Cross entropy loss requires one-hot output and a non-one-hot target with labels 0...N where N is the length
        # of the one-hot vectors
        loss = model.loss(output, masks)
        model.optimizer.zero_grad()
        loss.backward()
        model.optimizer.step()"""

# Final evaluation
test(verbose=True)
#accuracies.append(test(verbose=True))
#print('Final accuracy: ', accuracies[-1])
#plt.plot(range(len(accuracies)), accuracies)
