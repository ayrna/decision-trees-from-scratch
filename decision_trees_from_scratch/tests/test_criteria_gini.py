import numpy as np
from decision_trees_from_scratch._tree_split_criteria import Gini
from sklearn.utils import compute_sample_weight


def generate_array(counts):
    array = [num for num, count in counts.items() for _ in range(count)]
    return array


def test_gini():
    ## Test 1

    left_counts = {
        0: 97,
        1: 3,
        2: 1,
        3: 2,
    }
    left = generate_array(left_counts)

    right_counts = {
        0: 2,
        1: 85,
        2: 5,
        3: 6,
    }
    right = generate_array(right_counts)

    counts_target = {
        0: left_counts[0] + right_counts[0],
        1: left_counts[1] + right_counts[1],
        2: left_counts[2] + right_counts[2],
        3: left_counts[3] + right_counts[3],
    }
    target = generate_array(counts_target)

    criterion_val_p1, parent_impurity = Gini(n_classes=4).compute(target, left, right)

    assert np.isclose(criterion_val_p1, 0.388493145987579, atol=1e-8, rtol=1e-8)

    ## Test 2

    left_counts = {
        0: 45,
        1: 43,
        2: 34,
        3: 50,
    }
    left = generate_array(left_counts)

    right_counts = {
        0: 23,
        1: 31,
        2: 18,
        3: 19,
    }
    right = generate_array(right_counts)

    counts_target = {
        0: left_counts[0] + right_counts[0],
        1: left_counts[1] + right_counts[1],
        2: left_counts[2] + right_counts[2],
        3: left_counts[3] + right_counts[3],
    }
    target = generate_array(counts_target)

    criterion_val_p1, parent_impurity = Gini(n_classes=4).compute(target, left, right)

    assert np.isclose(criterion_val_p1, 0.00339580389807886, atol=1e-8, rtol=1e-8)


def test_gini_with_cw():

    ## Test 1

    left_counts = {
        0: 97,
        1: 3,
        2: 1,
        3: 2,
    }
    left = generate_array(left_counts)

    right_counts = {
        0: 2,
        1: 85,
        2: 5,
        3: 6,
    }
    right = generate_array(right_counts)

    counts_target = {
        0: left_counts[0] + right_counts[0],
        1: left_counts[1] + right_counts[1],
        2: left_counts[2] + right_counts[2],
        3: left_counts[3] + right_counts[3],
    }
    target = generate_array(counts_target)

    class_weights = {0: 0.61, 1: 2.3, 2: 1.421, 3: 0.98}
    sw = compute_sample_weight(class_weights, target)
    sw_left = compute_sample_weight(class_weights, left)
    sw_right = compute_sample_weight(class_weights, right)

    criterion_val_p1, parent_impurity = Gini(n_classes=4).compute(
        target, left, right, sw=sw, sw_left=sw_left, sw_right=sw_right
    )

    assert np.isclose(criterion_val_p1, 0.263498971716037, atol=1e-8, rtol=1e-8)

    ## Test 2

    left_counts = {
        0: 45,
        1: 43,
        2: 34,
        3: 50,
    }
    left = generate_array(left_counts)

    right_counts = {
        0: 23,
        1: 31,
        2: 18,
        3: 19,
    }
    right = generate_array(right_counts)

    counts_target = {
        0: left_counts[0] + right_counts[0],
        1: left_counts[1] + right_counts[1],
        2: left_counts[2] + right_counts[2],
        3: left_counts[3] + right_counts[3],
    }
    target = generate_array(counts_target)
    class_weights = {0: 0.23, 1: 1.32, 2: 2.7, 3: 1.2}
    sw = compute_sample_weight(class_weights, target)
    sw_left = compute_sample_weight(class_weights, left)
    sw_right = compute_sample_weight(class_weights, right)

    criterion_val_p1, parent_impurity = Gini(n_classes=4).compute(
        target, left, right, sw=sw, sw_left=sw_left, sw_right=sw_right
    )

    assert np.isclose(criterion_val_p1, 0.00325937379505328, atol=1e-8, rtol=1e-8)
