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

    def node_impurity(self, y, **kwargs):
        """
        Calculate the impurity of a single node depending on the criterion, e.g. the impurity of the Information
        Gain criterion is the entropy.

        Parameters:
        y (array-like): Array of output labels at the node.

        Returns:
        float: Impurity of the node.
        """
        return self._node_impurity(y, **kwargs)

    def compute(self, y, left_y, right_y, **kwargs):
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
        return self._compute(y, left_y, right_y, **kwargs)
