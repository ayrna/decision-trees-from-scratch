import numpy as np
from decision_trees_from_scratch._tree_split_criteria import InformationGain
from sklearn.utils import compute_sample_weight


def generate_array(counts):
    array = [num for num, count in counts.items() for _ in range(count)]
    return np.array(array)


def test_information_gain():

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

    criterion_val, parent_impurity = InformationGain(n_classes=4).compute(target, left, right, sample_weight=None)

    assert np.isclose(criterion_val, 0.783722817494008, atol=1e-8, rtol=1e-8)

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

    criterion_val, parent_impurity = InformationGain(n_classes=4).compute(
        target,
        left,
        right,
        sample_weight=None,
    )

    assert np.isclose(criterion_val, 0.00901662458328911, atol=1e-8, rtol=1e-8)


def test_information_gain_sample_weight():
    left_counts = {
        0: 78,
        1: 129,
        2: 343,
        3: 167,
    }
    left = generate_array(left_counts)

    right_counts = {
        0: 102,
        1: 312,
        2: 181,
        3: 43,
    }
    right = generate_array(right_counts)

    counts_target = {
        0: left_counts[0] + right_counts[0],
        1: left_counts[1] + right_counts[1],
        2: left_counts[2] + right_counts[2],
        3: left_counts[3] + right_counts[3],
    }
    target = generate_array(counts_target)

    class_weights = {0: 0.5, 1: 1.3, 2: 2.4, 3: 0.7}
    sw = compute_sample_weight(class_weights, target)
    sw_left = compute_sample_weight(class_weights, left)
    sw_right = compute_sample_weight(class_weights, right)

    criterion_val, parent_impurity = InformationGain(n_classes=4).compute(
        target, left, right, sw=sw, sw_left=sw_left, sw_right=sw_right
    )

    assert np.isclose(criterion_val, 0.0891720282003584, atol=1e-8, rtol=1e-8)

    left_counts = {
        0: 234,
        1: 121,
        2: 89,
        3: 99,
    }
    left = generate_array(left_counts)

    right_counts = {
        0: 123,
        1: 256,
        2: 189,
        3: 53,
    }
    right = generate_array(right_counts)

    counts_target = {
        0: left_counts[0] + right_counts[0],
        1: left_counts[1] + right_counts[1],
        2: left_counts[2] + right_counts[2],
        3: left_counts[3] + right_counts[3],
    }
    target = generate_array(counts_target)

    class_weights = {0: 0.5, 1: 1.3, 2: 2.4, 3: 0.7}
    sw = compute_sample_weight(class_weights, target)
    sw_left = compute_sample_weight(class_weights, left)
    sw_right = compute_sample_weight(class_weights, right)

    criterion_val, parent_impurity = InformationGain(n_classes=4).compute(
        target, left, right, sw=sw, sw_left=sw_left, sw_right=sw_right
    )

    assert np.isclose(criterion_val, 0.0524389087527378, atol=1e-8, rtol=1e-8)
