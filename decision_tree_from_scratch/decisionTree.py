import decision_tree_from_scratch.decisionTree_splitCriteria as criterias
from decision_tree_from_scratch.decisionTree_node import Node
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.utils.multiclass import unique_labels
from sklearn.utils.validation import check_array, check_is_fitted, check_X_y


class DTC(ClassifierMixin, BaseEstimator):
    def __init__(
        self,
        criterion="gini",
        depth=0,
        max_depth=5,
        random_state=None,
    ):
        self.criterion = criterion
        self.depth = depth
        self.max_depth = max_depth
        self.random_state = random_state

    def fit(self, X, y):
        # X, y = check_X_y(X, y)
        self.classes_ = unique_labels(y)

        if self.criterion == "information_gain":
            self._criterion = criterias.information_gain
        elif self.criterion == "gini":
            self._criterion = criterias.gini
        elif self.criterion == "gini_simpson_extended":
            self._criterion = criterias.gini_simpson_extended
        elif self.criterion == "ordinal_gini_simpson_extended":
            self._criterion = criterias.ordinal_gini_simpson_extended
        elif self.criterion == "weighted_information_gain":
            self._criterion = criterias.weighted_information_gain
        elif self.criterion == "ranking_impurity":
            self._criterion = criterias.ranking_impurity
        else:
            raise ValueError(f"Criterion {self.criterion} not recognized.")

        self._tree = Node(
            criterion=self._criterion,
            max_depth=self.max_depth,
            random_state=self.random_state,
        )

        self._tree.fit(X, y)

        return self

    def predict(self, X):
        check_is_fitted(self)
        # X = check_array(X)

        return self._tree.predict(X)
