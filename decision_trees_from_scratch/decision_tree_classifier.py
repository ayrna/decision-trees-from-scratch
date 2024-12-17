import numpy as np
import decision_trees_from_scratch._tree_split_criteria as criterias
from decision_trees_from_scratch._tree_ import Tree
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.utils.multiclass import unique_labels


class DTC(ClassifierMixin, BaseEstimator):
    def __init__(
        self,
        criterion="gini",
        max_depth=5,
        random_state=None,
    ):
        self.criterion = criterion
        self.max_depth = max_depth
        self.random_state = random_state

    def fit(self, X, y):
        self.classes_ = unique_labels(y)
        self.n_classes_ = max(self.classes_) + 1

        if not np.array_equal(self.classes_, np.arange(self.n_classes_)):
            raise ValueError("Classes should be labeled from 0 to n_classes - 1.")

        if self.criterion == "gini":
            self._criterion = criterias.Gini(n_classes=self.n_classes_)
        elif self.criterion == "ig":
            self._criterion = criterias.InformationGain(n_classes=self.n_classes_)
        elif self.criterion == "ogini":
            self._criterion = criterias.OrdinalGini(n_classes=self.n_classes_)
        elif self.criterion == "wig":
            self._criterion = criterias.WeightedInformationGain(n_classes=self.n_classes_, power=1)
        elif self.criterion == "ri":
            self._criterion = criterias.RankingImpurity(n_classes=self.n_classes_)
        else:
            raise ValueError(f"Criterion {self.criterion} not recognized.")

        self._tree = Tree(
            criterion=self._criterion,
            max_depth=self.max_depth,
            random_state=self.random_state,
        )

        self._tree.grow(X, y)

        return self

    def predict(self, X):
        return self._tree.predict(X)

    def predict_proba(self, X):
        return self._tree.predict_proba(X)

    def print_tree(self):
        self._tree.print_node_tree()

    def get_tree_depth(self):
        return self._tree.get_tree_depth()
