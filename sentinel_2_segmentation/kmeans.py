import numpy as np
import cv2 as cv
from sentinel_2_segmentation.utils import permuted_labels
from sentinel_2_segmentation.data import get_subscenes, get_masks

# Loading data
subscenes, subscenes_flattened = get_subscenes()
masks = get_masks()

# K-means clustering
for subscene, subscene_flattened, mask in zip(subscenes, subscenes_flattened, masks):
    criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 10, 1.0)
    compactness, labels, centers = cv.kmeans(subscene_flattened, 3, None, criteria, attempts=10,
                                             flags=cv.KMEANS_RANDOM_CENTERS)
    labels = labels.reshape(subscene.shape[0], subscene.shape[1])
    #labels = np.random.randint(0, 3, size=(1022, 1022))
    percentages = []
    for label_permutation in permuted_labels(labels):
        percentages.append(np.average(mask == label_permutation))
    print(max(percentages))
