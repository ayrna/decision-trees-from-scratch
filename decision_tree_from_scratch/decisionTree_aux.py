import numpy as np


## Function to compute the moving average to work with numeric features
#
def moving_average(feature_values):
    sorted_values = np.sort(feature_values)

    # Calculate midpoints between consecutive values
    midpoints = (sorted_values[:-1] + sorted_values[1:]) / 2.0

    return np.unique(midpoints)


## Function to evaluate a feature, it first extract a set of thresholds,
#  and then test each one, returning the best in terms of the criterion
#
#
def evaluate_feature(feature, target, criterion, random_state):
    best_threshold = None
    best_criterion_value = None

    if len(target) < 2:
        return best_threshold, best_criterion_value

    thresholds = moving_average(feature)
    for i, Xf in enumerate(thresholds):
        left_index = feature <= Xf
        left_target = target[left_index]
        right_target = target[~left_index]
        if (len(left_target) == 0) or (len(right_target) == 0):
            continue
        criterion_value = criterion(target, left_target, right_target)

        # results[i] = criterion_value
        if best_criterion_value is None:
            best_criterion_value = criterion_value
            best_threshold = Xf
        elif criterion_value > best_criterion_value:
            best_threshold = Xf
            best_criterion_value = criterion_value

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
    return best
