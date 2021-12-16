import os
import argparse
import time
import functools
from matplotlib import pyplot as plt
import torch
import torchvision
from torch.utils.data import DataLoader, random_split
import segmentation_models_pytorch as smp
from config import mask_dir, training_data
from neural_net import NeuralNet
from rasterio_dataset import RasterioDataset
from metrics import simple_pixel_accuracy, iou
from evaluator import Evaluator
from utils import get_rasters_from_file

# Command line arg processing
parser = argparse.ArgumentParser()
parser.add_argument('-s', '--model-save-path', type=str, nargs=1)
parser.add_argument('-l', '--model-load-path', type=str, nargs=1)
parser.add_argument('-b', '--bbox-coords', type=int, nargs=4)
args = parser.parse_args()
smp.unet.Unet()
# Parameters
test_set_size = 50
num_training_epochs = 10
test_epoch_period = 2
model_save_period = 2
model = NeuralNet(loss=smp.losses.JaccardLoss(mode='multiclass', classes=[1]))
if args.model_load_path:
    model.load(args.model_load_path[0])
iou_by_class = functools.partial(iou, reduce=False)
soft_iou = functools.partial(iou, soft=True)
soft_iou_by_class = functools.partial(iou, soft=True, reduce=False)
evaluator = Evaluator(['Pixel accuracy', 'IoU', 'IoU for negative class', 'IoU for positive class', 'Soft IoU',
                       'Soft IoU for negative class', 'Soft IoU for positive class'],
                      [simple_pixel_accuracy,
                       iou,
                       lambda pred, target: iou_by_class(pred, target)[0].item(),
                       lambda pred, target: iou_by_class(pred, target)[1].item(),
                       soft_iou,
                       lambda pred, target: soft_iou_by_class(pred, target)[0].item(),
                       lambda pred, target: soft_iou_by_class(pred, target)[1].item()
                       ])

# Load data
data = RasterioDataset(lambda: get_rasters_from_file(training_data), mask_dir, args.bbox_coords,
                       transform=torchvision.transforms.ToTensor())

# Split into training and testing sets and create data loaders
train_data, test_data = random_split(data, [len(data) - test_set_size, test_set_size])
train_loader = DataLoader(train_data, batch_size=1, shuffle=True)
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
for i in range(num_training_epochs + 1):
    if args.model_save_path and i % model_save_period == 0:
        model.save(args.model_save_path[0] + '_' + str(i), i)
    if i == num_training_epochs:
        break
    if i % test_epoch_period == 0:
        accuracies.append(test(verbose=True))
        print(i, accuracies[-1])
    for rasters, masks in train_loader:
        print("Start : %s" % time.ctime())
        output = model(rasters)
        # Cross entropy loss requires one-hot output and a non-one-hot target with labels 0...N-1 where N is the length
        # of the one-hot vectors
        loss = model.loss(output, masks)
        model.optimizer.zero_grad()
        loss.backward()
        model.optimizer.step()
        print("End : %s" % time.ctime() + "\n")

# Final evaluation
accuracies.append(test(verbose=True))
print('Final Accuracy')
print_accuracies(evaluator.metric_names, accuracies[-1])
plt.plot(range(len(accuracies)), accuracies)
plt.xlabel('Number of evaluations')
plt.ylabel('Accuracy')
plt.legend(evaluator.get_metric_names(), loc='upper right')
plt.show()
