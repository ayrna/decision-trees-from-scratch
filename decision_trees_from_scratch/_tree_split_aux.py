import numpy as np


## Function to compute the moving average to work with numeric features
#
def moving_average(feature_values):
    if len(np.unique(feature_values)) > 5:
        sorted_values = np.sort(feature_values)
        midpoints = (sorted_values[:-1] + sorted_values[1:]) / 2.0
        return np.unique(midpoints)
    else:
        return np.unique(feature_values)


class ClassDistribution:
    def __init__(self, y, Q, sample_weight=None):
        self.Q = Q
        labels, counts = np.unique(y, return_counts=True)
        f = {l: c for l, c in zip(labels, counts)}
        self.counts = np.ones(self.Q) * -1

        if sample_weight is None:
            for i in np.arange(self.Q):
                self.counts[i] = f[i] if i in f else 0
            self.counts = self.counts.astype(int)
        else:
            for i in np.arange(self.Q):
                self.counts[i] = np.sum(sample_weight[y == i]) if i in f else 0

        self.labels = np.arange(self.Q)

    def get_probas(self):
        return self.counts / np.sum(self.counts)

    def get_cumprobas(self):
        return np.cumsum(self.get_probas())

    def get_non_zero_labels(self):
        return np.nonzero(self.counts)[0]

    def get_non_zero_probas(self):
        return self.get_probas()[self.get_non_zero_labels()]

    def __repr__(self):
        print(self.counts)
