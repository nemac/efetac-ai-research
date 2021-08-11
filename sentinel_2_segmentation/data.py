import numpy as np
from sentinel_2_segmentation.utils import filepaths

# Parameters
num_data_points = 200


# Gets subscenes
def get_subscenes():
    subscene_directory = 'C:/Users/CyborgOctopus/Downloads/Sentinel 2 Cloud Mask Catalogue/subscenes/subscenes'
    subscene_paths = filepaths(subscene_directory)
    subscenes = [np.load(path) for path in subscene_paths[0:num_data_points]]
    rgb_subscenes = [subscene[..., [3, 2, 1]] for subscene in subscenes]
    rgb_subscenes_flattened = [subscene.reshape(subscene.shape[0] * subscene.shape[1], 3) for subscene in rgb_subscenes]
    return np.array(rgb_subscenes), np.array(rgb_subscenes_flattened)


# Gets masks
def get_masks():
    mask_directory = 'C:/Users/CyborgOctopus/Downloads/Sentinel 2 Cloud Mask Catalogue/masks/masks'
    mask_paths = filepaths(mask_directory)
    masks = [np.load(path) for path in mask_paths[0:num_data_points]]
    return np.array([mask.argmax(axis=len(mask.shape) - 1) for mask in masks])
