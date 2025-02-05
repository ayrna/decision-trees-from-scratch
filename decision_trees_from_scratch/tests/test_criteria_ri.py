import numpy as np
from decision_trees_from_scratch._tree_split_criteria import RankingImpurity
from sklearn.utils import compute_sample_weight


def generate_array(counts):
    array = [num for num, count in counts.items() for _ in range(count)]
    return array


def test_ranking_impurity():
    ## Test 1

    left_counts = {
        0: 28,
        1: 20,
        2: 1,
        3: 2,
    }
    left = generate_array(left_counts)

    right_counts = {
        0: 2,
        1: 5,
        2: 35,
        3: 67,
    }
    right = generate_array(right_counts)

    counts_target = {
        0: left_counts[0] + right_counts[0],
        1: left_counts[1] + right_counts[1],
        2: left_counts[2] + right_counts[2],
        3: left_counts[3] + right_counts[3],
    }
    target = generate_array(counts_target)

    criterion_val, parent_impurity = RankingImpurity(n_classes=4).compute(target, left, right)

    assert np.isclose(criterion_val, 11326, atol=1e-8, rtol=1e-8)

    ## Test 2

    left_counts = {
        0: 45,
        1: 3,
        2: 6,
        3: 89,
    }
    left = generate_array(left_counts)

    right_counts = {
        0: 2,
        1: 45,
        2: 43,
        3: 3,
    }
    right = generate_array(right_counts)

    counts_target = {
        0: left_counts[0] + right_counts[0],
        1: left_counts[1] + right_counts[1],
        2: left_counts[2] + right_counts[2],
        3: left_counts[3] + right_counts[3],
    }
    target = generate_array(counts_target)

    criterion_val, parent_impurity = RankingImpurity(n_classes=4).compute(target, left, right)

    assert np.isclose(criterion_val, 19136, atol=1e-8, rtol=1e-8)

    ## Test 3

    left_counts = {
        0: 45,
        1: 34,
        2: 14,
        3: 3,
    }
    left = generate_array(left_counts)

    right_counts = {
        0: 0,
        1: 56,
        2: 55,
        3: 45,
    }
    right = generate_array(right_counts)

    counts_target = {
        0: left_counts[0] + right_counts[0],
        1: left_counts[1] + right_counts[1],
        2: left_counts[2] + right_counts[2],
        3: left_counts[3] + right_counts[3],
    }
    target = generate_array(counts_target)

    criterion_val, parent_impurity = RankingImpurity(n_classes=4).compute(target, left, right)

    assert np.isclose(criterion_val, 20390, atol=1e-8, rtol=1e-8)


def test_ranking_impurity_with_sw():
    ## Test 1

    left_counts = {
        0: 28,
        1: 20,
        2: 1,
        3: 2,
    }
    left = generate_array(left_counts)

    right_counts = {
        0: 2,
        1: 5,
        2: 35,
        3: 67,
    }
    right = generate_array(right_counts)

    counts_target = {
        0: left_counts[0] + right_counts[0],
        1: left_counts[1] + right_counts[1],
        2: left_counts[2] + right_counts[2],
        3: left_counts[3] + right_counts[3],
    }
    target = generate_array(counts_target)

    class_weights = {0: 1.54, 1: 2.34, 2: 0.5, 3: 0.78}
    sw = compute_sample_weight(class_weights, target)
    sw_left = compute_sample_weight(class_weights, left)
    sw_right = compute_sample_weight(class_weights, right)

    criterion_val, parent_impurity = RankingImpurity(n_classes=4).compute(
        target, left, right, sw, sw_left, sw_right
    )

    assert np.isclose(criterion_val, 14742.016, atol=1e-8, rtol=1e-8)

    ## Test 2

    left_counts = {
        0: 45,
        1: 3,
        2: 6,
        3: 89,
    }
    left = generate_array(left_counts)

    right_counts = {
        0: 2,
        1: 45,
        2: 43,
        3: 3,
    }
    right = generate_array(right_counts)

    counts_target = {
        0: left_counts[0] + right_counts[0],
        1: left_counts[1] + right_counts[1],
        2: left_counts[2] + right_counts[2],
        3: left_counts[3] + right_counts[3],
    }
    target = generate_array(counts_target)

    class_weights = {0: 0.54, 1: 1.44, 2: 0.75, 3: 0.78}
    sw = compute_sample_weight(class_weights, target)
    sw_left = compute_sample_weight(class_weights, left)
    sw_right = compute_sample_weight(class_weights, right)

    criterion_val, parent_impurity = RankingImpurity(n_classes=4).compute(
        target, left, right, sw, sw_left, sw_right
    )

    assert np.isclose(criterion_val, 15249.177, atol=1e-8, rtol=1e-8)

    ## Test 3

    left_counts = {
        0: 45,
        1: 34,
        2: 14,
        3: 3,
    }
    left = generate_array(left_counts)

    right_counts = {
        0: 0,
        1: 56,
        2: 55,
        3: 45,
    }
    right = generate_array(right_counts)

    counts_target = {
        0: left_counts[0] + right_counts[0],
        1: left_counts[1] + right_counts[1],
        2: left_counts[2] + right_counts[2],
        3: left_counts[3] + right_counts[3],
    }
    target = generate_array(counts_target)

    class_weights = {0: 0.94, 1: 1.64, 2: 1.75, 3: 2.78}
    sw = compute_sample_weight(class_weights, target)
    sw_left = compute_sample_weight(class_weights, left)
    sw_right = compute_sample_weight(class_weights, right)

    criterion_val, parent_impurity = RankingImpurity(n_classes=4).compute(
        target, left, right, sw, sw_left, sw_right
    )

    assert np.isclose(criterion_val, 54870.4702, atol=1e-8, rtol=1e-8)
