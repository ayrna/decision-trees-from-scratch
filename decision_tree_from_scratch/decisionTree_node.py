import numpy as np
from decision_tree_from_scratch.decisionTree_aux import split


class Node:
    """
    CART implementation of a decision tree classifier.
    """

    def __init__(
        self,
        criterion,
        depth=0,
        max_depth=5,
        random_state=None,
        _root_y_classes=None,
        _root_y_count=None,
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

        if self.depth >= self.max_depth:
            self.leaf = True

    def fit(self, X, y):

        X = X.reset_index(drop=True)
        ## Save class distribution
        #
        #
        if self._root_y_classes is None:
            # We are in the root node
            y = y.to_numpy().astype(int)
            self._root_y_classes, self._root_y_count = np.unique(y, return_counts=True)

        #
        self.node_y_count = np.zeros(len(self._root_y_classes))
        for i, c in enumerate(self._root_y_classes):
            self.node_y_count[i] = len(np.where(y == c)[0])
        self.node_y = y
        ##

        ## Check if is a leaf, if not, fit if not fitted
        #
        #
        if self.leaf:
            self.fitted = True
            return
        #
        if not self.fitted:
            split_result = split(X, y, self.criterion, self.random_state)
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
        self.left = Node(
            depth=self.depth + 1,
            max_depth=self.max_depth,
            criterion=self.criterion,
            random_state=self.random_state,
            _root_y_classes=self._root_y_classes,
            _root_y_count=self._root_y_count,
        )
        self.left.fit(X_left, y_left)
        #
        self.right = Node(
            depth=self.depth + 1,
            max_depth=self.max_depth,
            criterion=self.criterion,
            random_state=self.random_state,
            _root_y_classes=self._root_y_classes,
            _root_y_count=self._root_y_count,
        )
        self.right.fit(X_right, y_right)
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

    #
    def predict(self, data):
        preds = []
        for i, entry in data.iterrows():
            preds.append(self.predict_entry(entry))
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

    def print_tree(self, side="left"):
        if self.leaf:
            print("\t" * self.depth, np.round(self.node_y_count, 2))
        else:
            self.print_node_info("left")
            self.left.print_tree("left")

            self.print_node_info("right")
            self.right.print_tree("right")
        return

    ##
