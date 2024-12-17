import numpy as np
from decision_trees_from_scratch._tree_split_ import split


class Tree:
    """
    CART implementation of a tree growth algorithm.
    """

    def __init__(
        self,
        criterion,
        depth=0,
        max_depth=5,
        random_state=None,
        _root_y_classes=None,
        _root_y_count=None,
        _root_y_probas=None,
    ):

        self.depth = depth
        self.max_depth = max_depth
        self.criterion = criterion

        self.feature = None
        self.threshold = None
        self.left = None
        self.right = None

        self.fitted = False
        self.leaf = False

        self.random_state = random_state

        self._root_y_classes = _root_y_classes
        self._root_y_count = _root_y_count
        self._root_y_probas = _root_y_probas

        if self.depth >= self.max_depth:
            self.leaf = True

    def grow(self, X, y):

        X = X.reset_index(drop=True)

        ## Save class distribution
        #
        #
        if self._root_y_classes is None:
            # We are in the root node
            y = y.to_numpy().astype(int)
            self._root_y_classes, self._root_y_count = np.unique(y, return_counts=True)
            self._root_y_probas = {c: p for c, p in zip(self._root_y_classes, self._root_y_count / len(y))}
        #
        self.node_y_count = np.zeros(len(self._root_y_classes))
        for i, c in enumerate(self._root_y_classes):
            self.node_y_count[i] = len(np.where(y == c)[0])
        self.node_y = y
        ##

        ## Check if this node is a leaf, if not, split
        #
        #
        if self.leaf:
            self.fitted = True
            return
        #
        if not self.fitted:
            split_result = split(
                X=X,
                y=y,
                root_y_probas=self._root_y_probas,
                criterion=self.criterion,
                random_state=self.random_state,
            )
            self.fitted = True

            if split_result is None:
                self.leaf = True
                return
            else:
                best_feature_name, _, best_threshold, criterion_val = split_result
                self.feature = best_feature_name
                self.threshold = best_threshold
                self.criterion_val = criterion_val
        ##

        ## Split the data and call recursively to child nodes
        #
        #
        left_indices = X[self.feature] <= self.threshold
        X_left = X[left_indices]
        y_left = y[left_indices]
        X_right = X[~left_indices]
        y_right = y[~left_indices]
        #
        self.left = Tree(
            depth=self.depth + 1,
            max_depth=self.max_depth,
            criterion=self.criterion,
            random_state=self.random_state,
            _root_y_classes=self._root_y_classes,
            _root_y_count=self._root_y_count,
            _root_y_probas=self._root_y_probas,
        )
        self.left.grow(X_left, y_left)
        #
        self.right = Tree(
            depth=self.depth + 1,
            max_depth=self.max_depth,
            criterion=self.criterion,
            random_state=self.random_state,
            _root_y_classes=self._root_y_classes,
            _root_y_count=self._root_y_count,
            _root_y_probas=self._root_y_probas,
        )
        self.right.grow(X_right, y_right)
        #
        ##

    ## Prediction functions, first a recursive single entry prediction function,
    # and then the final predict that receives a full dataframe
    #
    #
    def predict_entry(self, entry):
        if self.leaf:
            # What percentage of initial classes fall in this leaf
            # return self._root_y_classes[np.argmax(self.node_y_count / self._root_y_count)]
            return self._root_y_classes[np.argmax(self.node_y_count)]

        if entry[self.feature] <= self.threshold:
            return self.left.predict_entry(entry)
        else:
            return self.right.predict_entry(entry)

    def predict_entry_proba(self, entry):
        if self.leaf:
            # What percentage of initial classes fall in this leaf
            # return self._root_y_classes[np.argmax(self.node_y_count / self._root_y_count)]
            return self.node_y_count / len(self.node_y)

        if entry[self.feature] <= self.threshold:
            return self.left.predict_entry_proba(entry)
        else:
            return self.right.predict_entry_proba(entry)

    #
    def predict(self, data):
        preds = []
        for i, entry in data.iterrows():
            preds.append(self.predict_entry(entry))
        return preds

    def predict_proba(self, data):
        preds = []
        for i, entry in data.iterrows():
            preds.append(self.predict_entry_proba(entry))
        return preds

    ##

    ## Reset tree
    #
    #
    def reset_Tree(self):
        self.feature = None
        self.threshold = None
        self.left = None
        self.right = None

        self.fitted = False
        self.leaf = False
        self.y_classes = None
        self.y_distrib = None

        self.depth = 0

    # Print tree
    #
    #
    def print_node_info(self, side):
        if side == "left":
            print(
                "\t" * self.depth,
                "if",
                self.feature,
                "<=",
                round(self.threshold, 3),
            )
        elif side == "right":
            print(
                "\t" * self.depth,
                "elif",
                self.feature,
                ">",
                round(self.threshold, 3),
            )

    def print_node_tree(self):
        if self.leaf:
            print("\t" * self.depth, np.round(self.node_y_count, 2))
        else:
            self.print_node_info("left")
            self.left.print_node_tree()

            self.print_node_info("right")
            self.right.print_node_tree()
        return

    def get_tree_depth(self):
        if self.leaf:
            return self.depth
        else:
            return max(self.left.get_tree_depth(), self.right.get_tree_depth())

    ##
