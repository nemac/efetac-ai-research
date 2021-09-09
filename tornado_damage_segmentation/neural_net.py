import os
import torch
from torch import nn
from torch.nn import functional as F
from config import models_dir


class NeuralNet(nn.Module):
    """Defines a convolutional neural network"""

    def __init__(self, loss):
        super(NeuralNet, self).__init__()
        self.conv1 = nn.Conv2d(1, 10, 10)
        self.conv2 = nn.Conv2d(10, 2, 10)  # There are 2 possible states: tornado damage and no tornado damage
        self.loss = loss
        self.optimizer = torch.optim.Adam(self.parameters(), lr=1e-5)

    def forward(self, image):
        # Padding of 9 is applied before each conv layer due to the kernel size of 10
        out1 = F.relu(self.conv1(F.pad(image, (0, 9, 0, 9))))
        return self.conv2(F.pad(out1, (0, 9, 0, 9)))

    # Saves a model to the specified file name/path within the models directory
    # Based on this:
    # https://stackoverflow.com/questions/63655048/how-can-i-save-my-training-progress-in-pytorch-for-a-certain-batch-no
    def save(self, path, training_epoch):
        state = {'model': self.state_dict(), 'optimizer': self.optimizer.state_dict(), 'epoch': training_epoch}
        torch.save(state, os.path.join(models_dir, path))

    # Loads a model from the specified file name/path in the models directory and returns the training epoch
    # Based on
    # https://stackoverflow.com/questions/63655048/how-can-i-save-my-training-progress-in-pytorch-for-a-certain-batch-no
    def load(self, path):
        state = torch.load(os.path.join(models_dir, path))
        self.load_state_dict(state['model'])
        self.optimizer.load_state_dict(state['optimizer'])
        return state['epoch']
