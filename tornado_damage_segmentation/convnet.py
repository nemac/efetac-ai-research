# TODO implement training/test set separation
import numpy as np
import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader
from matplotlib import pyplot as plt
from tornado_damage_segmentation.data import raster_path, mask_path
from tornado_damage_segmentation.neural_net import NeuralNet
from tornado_damage_segmentation.rasterio_dataset import RasterioDataset

# Parameters
batch_size = 10
test_set_size = 50
num_training_epochs = 10
test_epoch_period = 2
model = NeuralNet()

# Split into training and testing sets
indices = range(len(rasters))
rng = np.random.default_rng()
test_set_indices = rng.choice(indices, test_set_size, replace=False)
training_set_indices = np.delete(indices, test_set_indices)

# Load data
training_data = RasterioDataset(raster_path, mask_path)
#test_data =
training_loader = DataLoader(training_data, batch_size, shuffle=True)
#test_loader =


# Tests the neural network, returning average accuracy
def test(verbose=False):
    accuracies = []
    for i in test_set_indices:
        test_raster, test_mask = rasters[i], masks[i]
        output = model(test_raster)
        accuracies.append(np.average(test_mask == output.argmax(dim=1)))
        if verbose:
            print(accuracies[-1])
        return np.average(accuracies)


# Train the neural network
accuracies = []
for i in range(num_training_epochs):
    if i % test_epoch_period == 0:
        accuracies.append(test())
        print(i, accuracies[-1])
        batch_indices = rng.choice(training_set_indices, batch_size, replace=False)
        batch_rasters, batch_masks = torch.tensor(rasters[batch_indices]), torch.tensor(masks[batch_indices])
        output = model(batch_rasters)
        # Cross entropy loss requires one-hot output and a non-one-hot target with labels 0...N where N is the length
        # of the one-hot vectors
        loss = model.loss(output, batch_masks)
        model.optimizer.zero_grad()
        loss.backward()
        model.optimizer.step()

# Final evaluation
accuracies.append(test(verbose=True))
print('Final accuracy: ', accuracies[-1])
plt.plot(range(len(accuracies)), accuracies)
