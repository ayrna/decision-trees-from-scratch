import numpy as np


### Weighted IG. (Fen Xia, Wensheng Zhang, and Jue Wang. 2006)
##
#
def _weighted_entropy_single(y, weights):
    # TODO
    _, class_counts = np.unique(y, return_counts=True)
    class_probabilities = class_counts / len(y)
    entropy = -np.sum(class_probabilities * np.log2(class_probabilities))
    return entropy  # / len(class_probabilities)  # / n_classes


def weighted_information_gain(y_father, y_left, y_right):

    mode_class = np.argmax(np.bincount(y_father))

    uni = np.unique(y_father)
    den = 0
    for u in uni:
        den += abs(u - mode_class)

    weights = dict()
    for u in uni:
        weights[u] = abs(u - mode_class) / den

    n_father = len(y_father)
    n_left = len(y_left)
    n_right = len(y_right)

    entropy_left = _weighted_entropy_single(y_left, weights=weights)
    entropy_right = _weighted_entropy_single(y_right, weights=weights)

    split_entropy = ((n_left / n_father) * entropy_left) + (
        (n_right / n_father) * entropy_right
    )

    return -split_entropy


#
# ·······································································


### Ranking Impurity. (Fen Xia, Wensheng Zhang, and Jue Wang. 2006)
##
#
def ranking_impurity_single(target):
    labels, counts = np.unique(target, return_counts=True)
    ri = 0
    for j in len(labels):
        for i in range(j):
            ri += (labels[j] - labels[i]) * counts[i] * counts[j]
    return ri


#
#
def ranking_impurity(target, left_target, right_target):
    ri_father = ranking_impurity_single(target)
    ri_left = ranking_impurity_single(left_target)
    ri_right = ranking_impurity_single(right_target)
    return ri_father - (ri_left + ri_right)


#
# ·······································································


### Twoing Criterion. (Raffaella Piccarreta. 2007.)
##
#
def twoing_criterion(parent_classes, left_classes, right_classes):
    # Calculate the number of samples in each node
    n_parent = len(parent_classes)
    n_left = len(left_classes)
    n_right = len(right_classes)

    # Calculate the proportions of samples in each node
    p_L = n_left / n_parent
    p_R = n_right / n_parent

    # Calculate the class distributions in each node
    def class_distribution(classes):
        distribution = {}
        for c in classes:
            if c in distribution:
                distribution[c] += 1
            else:
                distribution[c] = 1
        for c in distribution:
            distribution[c] /= len(classes)
        return distribution

    parent_dist = class_distribution(parent_classes)
    left_dist = class_distribution(left_classes)
    right_dist = class_distribution(right_classes)

    # Calculate the twoing criterion
    twoing_value = 0
    for c in parent_dist:
        pi_t_C1_L = left_dist.get(c, 0)
        pi_t_C1_R = right_dist.get(c, 0)
        twoing_value += (pi_t_C1_L - pi_t_C1_R) ** 2

    twoing_value *= 2 * p_L * p_R

    return twoing_value


#
# ·······································································


# def twoing_criterion(target, left_target, right_target):
#     n_parent = len(target)
#     n_left = len(left_target)
#     n_right = len(right_target)

#     p_L = n_left / n_parent
#     p_R = n_right / n_parent

#     _, parent_counts = np.unique(target, return_counts=True)
#     _, left_counts = np.unique(left_target, return_counts=True)
#     _, right_counts = np.unique(right_target, return_counts=True)

#     parent_dist = parent_counts / n_parent
#     left_dist = left_counts / n_left
#     right_dist = right_counts / n_right

#     twoing_value = 0
#     for c in parent_dist:
#         pi_t_C1_L = left_dist.get(c, 0)
#         pi_t_C1_R = right_dist.get(c, 0)
#         twoing_value += (pi_t_C1_L - pi_t_C1_R) ** 2

#     twoing_value *= 2 * p_L * p_R

#     return twoing_value


# GINI ################################################################################
#######################################################################################


def efficient_gini_index(target, left_target, right_target):
    total_samples = len(target)
    left_samples = len(left_target)
    right_samples = len(right_target)

    if total_samples == 0:
        return 0

    _, left_counts = np.unique(left_target, return_counts=True)
    left_probs = left_counts / left_samples
    _, right_counts = np.unique(right_target, return_counts=True)
    right_probs = right_counts / right_samples

    left_gini = 1 - np.sum(left_probs**2)
    right_gini = 1 - np.sum(right_probs**2)

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


# ENTROPY #############################################################################
#######################################################################################


def _entropy_single(y):
    _, class_counts = np.unique(y, return_counts=True)
    class_probabilities = class_counts / len(y)
    entropy = -np.sum(class_probabilities * np.log2(class_probabilities))
    return entropy  # / len(class_probabilities)  # / n_classes


def information_gain(y_father, y_left, y_right):
    n_father = len(y_father)
    n_left = len(y_left)
    n_right = len(y_right)

    entropy_left = _entropy_single(y_left)
    entropy_right = _entropy_single(y_right)

    split_entropy = ((n_left / n_father) * entropy_left) + (
        (n_right / n_father) * entropy_right
    )

    return -split_entropy


###################################3
###################################3
