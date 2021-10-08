import os
import sys
import time
from matplotlib import pyplot as plt
import torch
import torchvision
from torch.utils.data import DataLoader, random_split
import segmentation_models_pytorch as smp
from config import raster_dir, mask_dir, models_dir
from neural_net import NeuralNet
from rasterio_dataset import RasterioDataset
from metrics import simple_pixel_accuracy, iou
from evaluator import Evaluator

# Parameters
test_set_size = 50
num_training_epochs = 10
test_epoch_period = 2
model_save_period = 2
model = NeuralNet(loss=torch.nn.CrossEntropyLoss())
evaluator = Evaluator(['Pixel accuracy', 'IoU', 'IoU by class', 'Soft IoU', 'Soft IoU by class'],
                      [simple_pixel_accuracy, iou, lambda pred, target: iou(pred, target, reduce=False),
                       lambda pred, target: iou(pred, target, soft=True),
                       lambda pred, target: iou(pred, target, soft=True, reduce=False)])

# Load data
bbox_coords = [float(arg) for arg in sys.argv[1:]]
data = RasterioDataset(raster_dir, mask_dir, bbox_coords,
                       transform=torchvision.transforms.ToTensor())

# Split into training and testing sets and create data loaders
train_data, test_data = random_split(data, [len(data) - test_set_size, test_set_size])
train_loader = DataLoader(train_data, batch_size=10, shuffle=True)
test_loader = DataLoader(test_data)


# Takes a list of metric names and a list of accuracies (of the same length), with each accuracy at the same index as
# the corresponding metric, and prints them out in a nice format
def print_accuracies(metrics, accuracies):
    for metric, acc in zip(metrics, accuracies):
        print(metric + ': ' + str(acc))


# Tests the neural network, returning average accuracies
def test(verbose=False):
    model.eval()
    with torch.no_grad():
        for raster, mask in test_loader:
            start_time = time.ctime()
            output = model(raster)
            end_time = time.ctime()
            evaluator.add_data_point(output, mask)
            if verbose:
                print("Start : %s" % start_time)
                print("End : %s" % end_time)
                print_accuracies(evaluator.metric_names, evaluator.accuracies[-1])
    model.train()
    return evaluator.average_accuracies()


# Train the neural network
accuracies = []
for i in range(num_training_epochs):
    if i % test_epoch_period == 0:
        accuracies.append(test(verbose=True))
        print(i, accuracies[-1])
    if i % model_save_period == 0:
        model.save(os.path.join(models_dir, str(time.ctime())), i)
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
print('Final Accuracy')
print_accuracies(evaluator.metric_names, accuracies[-1])
plt.plot(range(len(accuracies)), accuracies)
