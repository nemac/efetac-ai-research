import os
import numpy as np


# Generates a list of filepaths in a given directory
def filepaths(directory):
    filenames = [file for _, _, file in os.walk(directory)][0]
    return [directory + '/' + name for name in filenames]


# Given an array containing integer labels ranging from 0 to 2, generates every permutation of [0, 1, 2]. For each
# permutation, reassigns each label value in the array accordingly, producing six different arrays.
def permuted_labels(labels):
    permutations = [[0, 1, 2], [0, 2, 1], [1, 0, 2], [1, 2, 0], [2, 0, 1], [2, 1, 0]]
    permuted_labels = []
    for permutation in permutations:
        flattened_labels = labels.reshape(-1)
        flattened_labels = np.array([permutation[i] for i in flattened_labels])
        permuted_labels.append(flattened_labels.reshape(labels.shape))
    return permuted_labels
