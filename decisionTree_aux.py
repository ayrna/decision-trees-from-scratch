from math import floor

import numpy as np
from decisionTree_splitCriteria import efficient_gini_index
from numpy import random

## Function to compute the moving average to work with numeric features
#
#
# def moving_average(feature):
#     return random.uniform(feature.min(), feature.max(), 1000)


# def moving_average(feature):
#     return feature.quantile([i / 1000 for i in range(1, 1000)]).values


# def moving_average(feature):
#     window_size = [2]
#     feature = np.unique(feature)
#     feature = np.sort(feature)
#     # values = list(feature.copy())
#     values = []
#     for ws in window_size:
#         i = 0
#         while i < len(feature):
#             values.append(np.average(feature[i : i + ws]))
#             i += 1
#     return np.unique(values)


def moving_average(feature_values):
    # Sort feature values
    sorted_values = np.sort(feature_values)

    # Calculate midpoints between consecutive values
    midpoints = (sorted_values[:-1] + sorted_values[1:]) / 2.0

    # Filter unique split points
    unique_thresholds = np.unique(midpoints)

    return unique_thresholds


# def moving_average(feature_values):
#     return np.percentile(feature_values, q=np.arange(25, 100, 25))


## Function to evaluate a feature, it first extract a set of thresholds,
#  and then test each one, returning the best in terms of the criterion
#
#
def evaluate_feature(feature, target, criterion, random_state):
    best_threshold = None
    best_criterion_value = None
    results = dict()

    if len(target) < 2:
        return best_threshold, best_criterion_value

    # feature = (
    #     feature.sample(frac=1, random_state=random_state).reset_index(drop=True).to_numpy()
    # )
    # is_binary = np.unique(feature).size == 2
    # if False:
    #     left_target = target[feature <= min(np.unique(feature))]
    #     right_target = target[feature > min(np.unique(feature))]
    #     if not (len(left_target) == 0 or len(right_target) == 0):
    #         best_criterion_value = criterion(target, left_target, right_target)
    #         best_threshold = 0
    #     results[0] = best_criterion_value
    # else:
    thresholds = moving_average(feature)
    # thresholds = np.unique(feature)
    # thresholds = np.sort(thresholds)
    # original_thresholds = thresholds.copy()
    # thresholds = np.linspace(min(thresholds), max(thresholds), len(thresholds) * 2)
    for i, Xf in enumerate(thresholds):
        left_index = feature <= Xf
        left_target = target[left_index]
        right_target = target[~left_index]
        if (len(left_target) == 0) or (len(right_target) == 0):
            continue
        if np.isclose(-0.0390011, Xf):
            a_target, a_left_target, a_right_target = target, left_target, right_target
        if np.isclose(-0.03480245553267495, Xf):
            b_target, b_left_target, b_right_target = target, left_target, right_target
        criterion_value = criterion(target, left_target, right_target)

        # results[i] = criterion_value
        if best_criterion_value is None:
            best_criterion_value = criterion_value
            best_threshold = Xf
        elif criterion_value > best_criterion_value:
            best_threshold = Xf
            # best_thresholds.append((Xf + thresholds[i - 1]) / 2)
            best_criterion_value = criterion_value
        # else:
        #     if improving:
        #         break
        # if thresholds[i - 1] != np.array(thresholds)[(thresholds < Xf)][-1]:
        #     pass
        # if len(original_thresholds) > 2:
        #     thresholds.append(
        #         (Xf + np.array(original_thresholds)[(original_thresholds < Xf)][-1]) / 2
        #     )
        # if len(thresholds) > 2:
        #     best_threshold = (Xf + thresholds[i - 1]) / 2
        # else:
        # else:
        #     break

        # mid_Xf = Xf
        # mid_criteria_value = criterion_value
        # while True:
        #     mid_Xf = (mid_Xf + thresholds[i - 1]) / 2

        #     left_target = target[feature <= mid_Xf]
        #     right_target = target[feature > mid_Xf]
        #     if (len(left_target) == 0) or (len(right_target) == 0):
        #         break
        #     mid_criteria_value = criterion(target, left_target, right_target)
        #     if mid_criteria_value > best_criterion_value:
        #         best_criterion_value = mid_criteria_value
        #         best_thresholds.append(mid_Xf)
        #     else:
        #         break

        # mid_Xf = Xf
        # mid_criteria_value = criterion_value
        # while True:
        #     mid_Xf = (mid_Xf + thresholds[i + 1]) / 2

        #     left_target = target[feature <= mid_Xf]
        #     right_target = target[feature > mid_Xf]
        #     if (len(left_target) == 0) or (len(right_target) == 0):
        #         break
        #     mid_criteria_value = criterion(target, left_target, right_target)
        #     if mid_criteria_value > best_criterion_value:
        #         best_criterion_value = mid_criteria_value
        #         best_thresholds.append(mid_Xf)
        #     else:
        #         break

    # results = sorted(results.items(), key=lambda x: x[1])
    return best_threshold, best_criterion_value


## Get best split parameters based on given data and criterion, steps:
#   1. Iterate over features and find the best in terms of the given criterion
#   2. Get the best feature, returning its name, values, threshold and criterion value obtained
#
#
def split(X, y, criterion, random_state):
    best = None

    # 1.
    #
    for f_name, f_vals in X.items():
        threshold, criterion_val = evaluate_feature(f_vals, y, criterion, random_state)

        if threshold is None:
            continue

        # 2.
        #
        if best is None:
            best = [f_name, f_vals, threshold, criterion_val]
        elif criterion_val > best[3]:
            best = [f_name, f_vals, threshold, criterion_val]
    #
    _, y_counts = np.unique(y, return_counts=True)
    _, y_left_counts = np.unique(y[best[1] <= best[2]], return_counts=True)
    _, y_right_counts = np.unique(y[best[1] > best[2]], return_counts=True)
    return best
