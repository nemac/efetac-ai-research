import sys, time, torch, torchvision
import numpy as np
from torch.utils.data import DataLoader, random_split
import torch.nn.functional as F
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
bbox_coords = [float(arg) for arg in sys.argv[1:]]
data = RasterioDataset(raster_dir, mask_dir, bbox_coords,
                       transform=torchvision.transforms.ToTensor())

# Split into training and testing sets and create data loaders
train_data, test_data = random_split(data, [len(data) - test_set_size, test_set_size])
train_loader = DataLoader(train_data, batch_size=10, shuffle=True)
test_loader = DataLoader(test_data)


# Implements the Intersection over Union (IoU) or Jaccard metric
# Assumes raw logits rather than probabilities (this matters only for 'soft' mode)
# Based on the implementation here:
# https://github.com/qubvel/segmentation_models.pytorch/blob/master/segmentation_models_pytorch/losses/jaccard.py
# and here:
# https://github.com/qubvel/segmentation_models.pytorch/blob/master/segmentation_models_pytorch/losses/_functional.py
# (look for 'soft_jaccard_score')
#
# Also takes inspiration from https://torchmetrics.readthedocs.io/en/stable/references/modules.html?highlight=IoU#iou
def iou(prediction, target, soft=False, reduce=True, zero_over_zero_score=1):
    # N, C, *dims --> C, N, *dims --> C, N x product(*dims)
    pred_positives = prediction.transpose(0, 1).view(prediction.shape[1], -1)
    # N, *dims --> N x product(*dims) --> N x product(*dims), C --> C, N x product(*dims)
    true_positives = F.one_hot(target.view(-1), num_classes=prediction.shape[1]).transpose(0, 1)

    if soft:
        pred_positives = pred_positives.softmax(dim=0)
    else:
        # C, N x product(*dims) --> N x product(*dims) --> N x product(*dims), C --> C, N x product(*dims)
        pred_positives = F.one_hot(pred_positives.argmax(axis=0)).transpose(0, 1)

    intersection = torch.sum(pred_positives * true_positives, dim=1)
    union = torch.sum(pred_positives + true_positives, dim=1) - intersection
    result = intersection / union
    result[union == 0] = zero_over_zero_score

    if reduce:
        return result.mean().item()
    return result


# Tests the neural network, returning average accuracies
def test(verbose=False):
    accuracies = []
    model.eval()
    with torch.no_grad():
        for raster, mask in test_loader:
            start_time = time.ctime()
            output = model(raster)
            end_time = time.ctime()
            accuracies.append({
                'Pixel accuracy': np.average(output.argmax(axis=1) == mask),
                'IoU': iou(output, mask),
                'IoU by class': iou(output, mask, reduce=False),
                'Soft IoU': iou(output, mask, soft=True),
                'Soft IoU by class': iou(output, mask, soft=True, reduce=False)
            })
            if verbose:
                print("Start : %s" % start_time)
                print("End : %s" % end_time)
                for i, (key, value) in enumerate(accuracies[-1].items()):
                    print(key + ': ', value)
    model.train()
    return [np.mean([accuracies[i][key] for i in range(len(accuracies))]) for key in accuracies[-1].keys()]


# Train the neural network
accuracies = []
for i in range(num_training_epochs):
    if i % test_epoch_period == 0:
        accuracies.append(test(verbose=True))
        print(i, accuracies[-1])
    for rasters, masks in train_loader:
        print("Start : %s" % time.ctime())
        print('raster shape: ', rasters.shape)
        print('mask shape: ', masks.shape)
        output = model(rasters)
        print('output shape: ', output.shape)
        # Cross entropy loss requires one-hot output and a non-one-hot target with labels 0...N-1 where N is the length
        # of the one-hot vectors
        loss = model.loss(output, masks)
        model.optimizer.zero_grad()
        loss.backward()
        model.optimizer.step()
        print("End : %s" % time.ctime() + "\n")

# Final evaluation
#test(verbose=True)
accuracies.append(test(verbose=True))
print('Final accuracy: ', accuracies[-1])
plt.plot(range(len(accuracies)), accuracies)
