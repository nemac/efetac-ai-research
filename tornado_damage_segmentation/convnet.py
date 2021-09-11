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
def iou(prediction, target, soft=False):
    # N, C, *dims --> C, N, *dims --> C, N x product(*dims)
    pred_positives = prediction.transpose(0, 1).view(prediction.shape[1], -1)
    if soft:
        pred_positives = pred_positives.softmax(dim=0)
    else:
        # C, N x product(*dims) --> N x product(*dims) --> N x product(*dims), C --> C, N x product(*dims)
        pred_positives = F.one_hot(pred_positives.argmax(axis=0)).transpose(0, 1)
    # N, *dims --> N x product(*dims) --> N x product(*dims), C --> C, N x product(*dims)
    true_positives = F.one_hot(target.view(-1), num_classes=prediction.shape[1]).transpose(0, 1)
    intersection = torch.sum(pred_positives * true_positives, dim=1)
    union = torch.sum(pred_positives + true_positives, dim=1) - intersection
    return (intersection / union).mean().item()


# Tests the neural network, returning average accuracies
# TODO update accuracy measure.
def test(verbose=False):
    pixel_accuracies = []
    iou_accuracies = []
    my_iou_accuracies = []
    my_soft_iou_accuracies = []
    model.eval()
    with torch.no_grad():
        for raster, mask in test_loader:
            print("Start : %s" % time.ctime())
            print("raster shape: " + str(raster.shape))
            print("mask shape: " + str(mask.shape))
            output = model(raster)
            print("output shape: " + str(output.shape))
            pixel_accuracies.append(np.average(output.argmax(axis=1) == mask))
            iou_accuracies.append(1 - smp.losses.JaccardLoss(mode='multiclass').forward(output, mask).item())
            my_iou_accuracies.append(iou(output, mask))
            my_soft_iou_accuracies.append(iou(output, mask, soft=True))
            if verbose:
                print('Pixel Accuracy: ', pixel_accuracies[-1])
                print('IoU: ', iou_accuracies[-1])
                print('My IoU: ', my_iou_accuracies[-1])
                print('My Soft IoU: ', my_soft_iou_accuracies[-1])
            print("End : %s" % time.ctime() + "\n")
    model.train()
    return {'pixel accuracy': np.mean(pixel_accuracies), 'iou': np.mean(iou_accuracies), 'my_iou': np.mean(my_iou_accuracies)}


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
