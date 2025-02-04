import numpy as np
import pandas as pd
from decision_trees_from_scratch._tree_ import Tree
from decision_trees_from_scratch._tree_split_criteria import OrdinalGini


def test_node_split_tracking_of_root_y_info():
    criterion = OrdinalGini(n_classes=4)
    tree = Tree(
        criterion=criterion,
        depth=0,
        max_depth=1,
        random_state=None,
        _root_y_classes=None,
        _root_y_count=None,
        _root_y_probas=None,
    )

    X = pd.DataFrame(
        [
            [1, 2],
            [3, 4],
            [5, 6],
            [7, 8],
            [9, 10],
            [11, 12],
            [13, 14],
            [15, 16],
            [17, 18],
            [19, 20],
        ]
    )
    y = pd.Series([0, 0, 0, 2, 2, 1, 3, 1, 1, 1])

    tree.grow(X, y, sample_weight=None)

    assert np.allclose(tree._root_y_classes, np.array([0, 1, 2, 3]))
    assert np.allclose(tree.left._root_y_classes, np.array([0, 1, 2, 3]))
    assert np.allclose(tree.right._root_y_classes, np.array([0, 1, 2, 3]))
    assert np.allclose(tree._root_y_count, np.array([3, 4, 2, 1]))
    assert np.allclose(tree.left._root_y_count, np.array([3, 4, 2, 1]))
    assert np.allclose(tree.right._root_y_count, np.array([3, 4, 2, 1]))
    assert tree._root_y_probas == {0: 0.3, 1: 0.4, 2: 0.2, 3: 0.1}
    assert tree.left._root_y_probas == {0: 0.3, 1: 0.4, 2: 0.2, 3: 0.1}
    assert tree.right._root_y_probas == {0: 0.3, 1: 0.4, 2: 0.2, 3: 0.1}


test_node_split_tracking_of_root_y_info()
