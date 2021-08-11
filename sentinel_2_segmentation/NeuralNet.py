import torch
from torch import nn
from torch.nn import functional as F


# Defines a convolutional neural network
class NeuralNet(nn.Module):
    def __init__(self):
        super(NeuralNet, self).__init__()
        self.conv1 = nn.Conv2d(3, 20, 10)
        self.conv2 = nn.Conv2d(20, 3, 10)
        self.loss = nn.CrossEntropyLoss()
        self.optimizer = torch.optim.Adam(self.parameters(), lr=1e-5)

    def forward(self, image):
        return F.softmax(self.conv2(F.relu(self.conv1(image))), dim=1)
