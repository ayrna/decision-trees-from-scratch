import pandas as pd
import numpy as np
from decision_tree_from_scratch.tree_node import Tree
from decision_tree_from_scratch.tree_split_criteria import OrdinalGini


def test_node_split_tracking_of_root_y_info():
    criterion = OrdinalGini(n_classes=4)
    node = Tree(
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

    node.fit(X, y)

    assert np.allclose(node._root_y_classes, np.array([0, 1, 2, 3]))
    assert np.allclose(node.left._root_y_classes, np.array([0, 1, 2, 3]))
    assert np.allclose(node.right._root_y_classes, np.array([0, 1, 2, 3]))
    assert np.allclose(node._root_y_count, np.array([3, 4, 2, 1]))
    assert np.allclose(node.left._root_y_count, np.array([3, 4, 2, 1]))
    assert np.allclose(node.right._root_y_count, np.array([3, 4, 2, 1]))
    assert node._root_y_probas == {0: 0.3, 1: 0.4, 2: 0.2, 3: 0.1}
    assert node.left._root_y_probas == {0: 0.3, 1: 0.4, 2: 0.2, 3: 0.1}
    assert node.right._root_y_probas == {0: 0.3, 1: 0.4, 2: 0.2, 3: 0.1}
