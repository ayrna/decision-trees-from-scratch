import math

import numpy as np

### Gini-Simpson. (Raffaella Piccarreta. 2007.)
##
#


def gini_simpson_extended(target, left_target, right_target):
    labels, counts = np.unique(target, return_counts=True)
    labels_L, counts_L = np.unique(left_target, return_counts=True)
    labels_R, counts_R = np.unique(right_target, return_counts=True)

    f = {l: c for l, c in zip(labels, counts)}
    f_L = {l: c for l, c in zip(labels_L, counts_L)}
    f_R = {l: c for l, c in zip(labels_R, counts_R)}

    p = {}
    p_L = {}
    p_R = {}
    for l, c in f.items():
        p[l] = c / len(target)
        p_L[l] = f_L[l] / len(left_target) if l in labels_L else 0
        p_R[l] = f_R[l] / len(right_target) if l in labels_R else 0

    PL = len(left_target) / len(target)
    PR = len(right_target) / len(target)

    gini = 0
    for l in labels:
        gini += np.power(p_L[l] - p_R[l], 2)

    return PL * PR * gini


def gini_simpson(target, left_target, right_target):
    p_L = len(left_target) / len(target)
    p_R = len(right_target) / len(target)

    impurity_F = _impurity(target)
    impurity_L = _impurity(left_target)
    impurity_R = _impurity(right_target)

    return impurity_F - (p_L * impurity_L) - (p_R * impurity_R)


def _impurity(target):
    _, counts = np.unique(target, return_counts=True)
    probs = counts / len(target)
    impurity = 1 - np.sum(probs**2)
    return impurity


#
##
### ·······································································


### Ordinal Gini-Simpson. (Raffaella Piccarreta. 2007.)
##
#
def ordinal_gini_simpson_extended(target, left_target, right_target):
    labels, counts = np.unique(target, return_counts=True)
    labels_L, counts_L = np.unique(left_target, return_counts=True)
    labels_R, counts_R = np.unique(right_target, return_counts=True)

    f = {l: c for l, c in zip(labels, counts)}
    f_L = {l: c for l, c in zip(labels_L, counts_L)}
    f_R = {l: c for l, c in zip(labels_R, counts_R)}

    for l in f:
        if l not in f_L:
            f_L[l] = 0
        if l not in f_R:
            f_R[l] = 0

    p_L = np.cumsum([*f_L.values()]) / len(left_target)
    p_R = np.cumsum([*f_R.values()]) / len(right_target)

    PL = len(left_target) / len(target)
    PR = len(right_target) / len(target)

    gini = 0
    for p_L_i, p_R_i in zip(p_L, p_R):
        gini += np.power(p_L_i - p_R_i, 2)

    return PL * PR * gini


#
##
### ·······································································


### Weighted IG. (Fen Xia, Wensheng Zhang, and Jue Wang. 2006)
##
#
def _weighted_entropy_single(y, weights):
    uni, class_counts = np.unique(y, return_counts=True)
    class_probabilities = class_counts / len(y)
    w = np.array([weights[u] for u in uni])
    entropy = -np.sum(w * class_probabilities * np.log2(class_probabilities))
    return entropy


def weighted_information_gain(y_father, y_left, y_right):

    mode_class = np.argmax(np.bincount(y_father))
    uni = np.unique(y_father)
    weight_denominator = np.sum(np.abs(uni - mode_class))
    weights = {
        u: (
            np.power(abs(u - mode_class), 2) / np.power(weight_denominator, 2)
            if not math.isclose(weight_denominator, 0.0)
            else 0
        )
        for u in uni
    }

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
##
### ·······································································


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
    n_parent = len(parent_classes)
    n_left = len(left_classes)
    n_right = len(right_classes)

    p_L = n_left / n_parent
    p_R = n_right / n_parent

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
