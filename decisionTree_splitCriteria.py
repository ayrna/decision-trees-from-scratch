import numpy as np

# GINI ############################################################################################
###################################################################################################


def efficient_gini_index(target, left_target, right_target):
    total_samples = len(target)
    left_samples = len(left_target)
    right_samples = len(right_target)

    if total_samples == 0:
        return 0

    # Calculate class probabilities for left node
    _, left_counts = np.unique(left_target, return_counts=True)
    left_probs = left_counts / left_samples

    # Calculate class probabilities for right node
    _, right_counts = np.unique(right_target, return_counts=True)
    right_probs = right_counts / right_samples

    # Calculate Gini impurity for left node
    left_gini = 1 - np.sum(left_probs**2)

    # Calculate Gini impurity for right node
    right_gini = 1 - np.sum(right_probs**2)

    # Weighted sum of Gini impurities
    weighted_gini = (left_samples / total_samples) * left_gini + (
        right_samples / total_samples
    ) * right_gini

    return weighted_gini


def gini(target, left_target, right_target):

    _, left_target_counts = np.unique(left_target, return_counts=True)
    _, right_target_counts = np.unique(right_target, return_counts=True)

    GI = 0
    n = len(target)

    n_left = len(left_target)
    left_counts = 0
    for n_i in left_target_counts:
        left_counts += (n_i / n_left) ** 2
    GI += (n_left / n) * (1 - left_counts)

    n_right = len(right_target)
    right_counts = 0
    for n_i in right_target_counts:
        right_counts += (n_i / n_right) ** 2
    GI += (n_right / n) * (1 - right_counts)

    return GI


# ENTROPY #########################################################################################
###################################################################################################


def _entropy_single(y):
    _, class_counts = np.unique(y, return_counts=True)
    class_probabilities = class_counts / len(y)
    entropy = -np.sum(class_probabilities * np.log2(class_probabilities))
    return entropy  # / len(class_probabilities)  # / n_classes


def information_gain(y_father, y_left, y_right):
    n_father = len(y_father)
    n_left = len(y_left)
    n_right = len(y_right)

    # Calculate entropies
    entropy_left = _entropy_single(y_left)
    entropy_right = _entropy_single(y_right)

    # Calculate weighted average entropy
    split_entropy = ((n_left / n_father) * entropy_left) + (
        (n_right / n_father) * entropy_right
    )

    # return entropy_father - split_entropy
    return -split_entropy


###################################3
###################################3
