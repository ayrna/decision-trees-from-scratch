from decision_trees_from_scratch._tree_split_aux import moving_average


## Function to evaluate a feature, it first extract a set of thresholds,
#  and then test each one, returning the best in terms of the criterion
#
#
def evaluate_feature(feature, target, criterion, sample_weight, random_state):
    best_threshold = None
    best_criterion_value = None

    if len(target) < 2:
        return best_threshold, best_criterion_value

    thresholds = moving_average(feature)
    for Xf in thresholds:
        left_index = feature <= Xf
        left_target = target[left_index]
        right_target = target[~left_index]
        if sample_weight is not None:
            left_sw = sample_weight[left_index]
            right_sw = sample_weight[~left_index]
        else:
            left_sw = None
            right_sw = None
        if (len(left_target) == 0) or (len(right_target) == 0):
            continue
        criterion_value = criterion.compute(
            y=target,
            y_left=left_target,
            y_right=right_target,
            sw=sample_weight,
            sw_left=left_sw,
            sw_right=right_sw,
        )

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
def split(X, y, criterion, sample_weight, random_state):
    best = None

    # 1.
    #
    for f_name, f_vals in X.items():
        threshold, criterion_val = evaluate_feature(f_vals, y, criterion, sample_weight, random_state)

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
