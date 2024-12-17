import numpy as np
from decision_trees_from_scratch._tree_split_criteria import WeightedInformationGain


def generate_array(counts):
    array = [num for num, count in counts.items() for _ in range(count)]
    return array


def test_weighted_information_gain():

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

    assert np.isclose(
        WeightedInformationGain(n_classes=4, power=1).compute(target, left, right), 0.0770388176848689, atol=1e-08
    )
    assert np.isclose(
        WeightedInformationGain(n_classes=4, power=2).compute(target, left, right), 0.0408551950243498, atol=1e-08
    )
    assert np.isclose(
        WeightedInformationGain(n_classes=4, power=4).compute(target, left, right), 0.0140813937855637, atol=1e-08
    )

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

    assert np.isclose(
        WeightedInformationGain(n_classes=4, power=1).compute(target, left, right), 0.0050607132246584, atol=1e-08
    )
    assert np.isclose(
        WeightedInformationGain(n_classes=4, power=2).compute(target, left, right), 0.00656088265670002, atol=1e-08
    )
    assert np.isclose(
        WeightedInformationGain(n_classes=4, power=4).compute(target, left, right), 0.0102086507681084, atol=1e-08
    )
