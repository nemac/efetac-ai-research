import sys, time, torch, torchvision
import numpy as np
from torch.utils.data import DataLoader, random_split
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


# Tests the neural network, returning average accuracies
# TODO update accuracy measure.
def test(verbose=False):
    pixel_accuracies = []
    iou_accuracies = []
    model.eval()
    with torch.no_grad():
        for raster, mask in test_loader:
            print("Start : %s" % time.ctime())
            print("raster shape: " + str(raster.shape))
            print("mask shape: " + str(mask.shape))
            output = model(raster)
            print("output shape: " + str(output.shape))
            pixel_accuracies.append(np.mean((output.argmax(axis=1) == mask).numpy()))
            iou_accuracies.append(1 - smp.losses.JaccardLoss(mode='multiclass').forward(output, mask).item())
            if verbose:
                print('Pixel Accuracy: ', pixel_accuracies[-1])
                print('IoU: ', iou_accuracies[-1])
            print("End : %s" % time.ctime() + "\n")
    model.train()
    return {'pixel accuracy': np.mean(pixel_accuracies), 'iou': np.mean(iou_accuracies)}


# Train the neural network
accuracies = []
for i in range(num_training_epochs):
    if i % test_epoch_period == 0:
        accuracies.append(test())
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
