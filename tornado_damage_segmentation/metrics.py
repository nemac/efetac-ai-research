# Defines metrics for use in evaluating the performance of the neural network
import numpy as np
import torch
import torch.nn.functional as F


# Returns the fraction of pixels which are classified correctly
def simple_pixel_accuracy(prediction, target):
    return np.average(prediction.argmax(axis=1) == target)


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
        pred_positives = F.one_hot(pred_positives.argmax(axis=0), num_classes=prediction.shape[1]).transpose(0, 1)

    intersection = torch.sum(pred_positives * true_positives, dim=1)
    union = torch.sum(pred_positives + true_positives, dim=1) - intersection
    result = intersection / union
    result[union == 0] = zero_over_zero_score

    if reduce:
        return result.mean().item()
    return result
