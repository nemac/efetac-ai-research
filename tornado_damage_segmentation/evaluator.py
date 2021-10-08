import numpy as np


class Evaluator:
    """Allows various metrics for accuracy to be defined and evaluated on data. A list of metric names and a list of
    functions defining the metrics must be passed. Data in the form of (prediction, target) pairs can then be added
    and evaluated based on these metrics."""

    def __init__(self, metric_names, metrics):
        self.metric_names = metric_names
        self.metrics = metrics
        self.data_points = []
        self.accuracies = []

    def add_data_point(self, prediction, target):
        #self.data_points.append((prediction, target))
        self.accuracies.append(map(lambda metric: metric(prediction, target), self.metrics))

    def average_accuracies(self):
        return np.mean(np.transpose(self.accuracies), axis=1).tolist()
