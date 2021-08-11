import numpy as np
import matplotlib.pyplot as plt
import torch
from sentinel_2_segmentation.data import get_subscenes, get_masks
from sentinel_2_segmentation.NeuralNet import NeuralNet
import torch.nn.functional as F

# Parameters
batch_size = 10
num_training_epochs = 100
test_epoch_period = 10
model = NeuralNet()

# Loading data
subscenes, _ = get_subscenes()
masks = get_masks()
subscenes = np.array([subscene.transpose(2, 0, 1) for subscene in subscenes])  # Switch channels from last to first axis
test_set_size = 50

# Split data into training and testing sets
indices = range(len(subscenes))
test_set_indices = np.random.choice(indices, test_set_size, replace=False)
training_set_indices = np.delete(indices, test_set_indices)
print(test_set_indices)
print(training_set_indices)


# Tests the neural network, returning average accuracy
def test(verbose=False):
    accuracies = []
    for i in test_set_indices:
        test_subscene, test_mask = torch.tensor([subscenes[i]]), torch.tensor([masks[i]])
        test_subscene = F.pad(test_subscene, pad=(0, 18, 0, 18))
        output = model(test_subscene)
        accuracies.append(np.average(test_mask == output.argmax(dim=1)))
        if verbose:
            print(accuracies[-1])
    return np.average(accuracies)


# Training
accuracies = []
for i in range(num_training_epochs):
    if i % test_epoch_period == 0:
        accuracies.append(test())
        print(i, accuracies[-1])
    batch_indices = np.random.choice(training_set_indices, batch_size, replace=False)
    batch_subscenes, batch_masks = torch.tensor(subscenes[batch_indices]), torch.tensor(masks[batch_indices])
    batch_subscenes = F.pad(batch_subscenes, pad=(0, 18, 0, 18))
    output = model(batch_subscenes)
    # Cross entropy loss requires one-hot output and a non-one-hot target with labels 0...N where N is the length of
    # the one-hot vectors (so confusing)
    loss = model.loss(output, batch_masks)
    model.optimizer.zero_grad()
    loss.backward()
    model.optimizer.step()

# Final evaluation
accuracies.append(test(verbose=True))
print('Final accuracy: ', accuracies[-1])
plt.plot(range(len(accuracies)), accuracies)
