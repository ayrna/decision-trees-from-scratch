import numpy as np


class SplitCriterion:
    """
    Split criterion base class.

    This class is the base class for all split criteria. It defines the interface that all split criteria must
    implement.

    Parameters:
    -----------
    n_classes (int): Number of classes in the target variable (at the beggining of the tree growth phase).
    """

    def __init__(self, n_classes):
        self.n_classes = n_classes

    def _compute_node_weights(self, y, y_left, y_right, sw, sw_left, sw_right):
        if sw is None:
            weight_node_left = len(y_left) / len(y)
            weight_node_right = len(y_right) / len(y)
        else:
            weight_node_left = np.sum(sw_left) / np.sum(sw)
            weight_node_right = np.sum(sw_right) / np.sum(sw)
        return weight_node_left, weight_node_right

    def node_impurity(self, y, sample_weight, **kwargs):
        """
        Calculate the impurity of a single node depending on the criterion, e.g. the impurity of the Information
        Gain criterion is the entropy.

        Parameters:
        y (array-like): Array of output labels at the node.

        Returns:
        float: Impurity of the node.
        """
        return self._node_impurity(y, sample_weight=sample_weight, **kwargs)

    def compute(self, y, y_left, y_right, sw=None, sw_left=None, sw_right=None, **kwargs):
        """
        Compute the final value of the split criterion. This is the decrease in impurity that results from the
        split.

        Parameters:
        y (array-like): Array of output labels at the parent node.
        left_y (array-like): Array of output labels at the left child node.
        right_y (array-like): Array of output labels at the right child node.

        Returns:
        float: Split criterion value.
        """
        return self._compute(y, y_left, y_right, sw, sw_left, sw_right, **kwargs)
