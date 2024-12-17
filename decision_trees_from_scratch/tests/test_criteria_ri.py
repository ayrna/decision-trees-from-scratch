import numpy as np
from decision_trees_from_scratch._tree_split_criteria import RankingImpurity


def generate_array(counts):
    array = [num for num, count in counts.items() for _ in range(count)]
    return array


def test_ranking_impurity():

    root_counts = {
        0: 345,
        1: 467,
        2: 290,
        3: 172,
    }
    root_target = generate_array(root_counts)

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

    criterion_val = RankingImpurity(n_classes=4).compute(
        target,
        left,
        right,
        root_y_probas={l: p / len(root_target) for l, p in root_counts.items()},
    )

    assert np.isclose(criterion_val, 11326, atol=1e-8)

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

    criterion_val = RankingImpurity(n_classes=4).compute(
        target,
        left,
        right,
        root_y_probas={l: p / len(root_target) for l, p in root_counts.items()},
    )

    assert np.isclose(criterion_val, 19136, atol=1e-8)

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

    criterion_val = RankingImpurity(n_classes=4).compute(
        target,
        left,
        right,
        root_y_probas={l: p / len(root_target) for l, p in root_counts.items()},
    )

    assert np.isclose(criterion_val, 20390, atol=1e-8)
