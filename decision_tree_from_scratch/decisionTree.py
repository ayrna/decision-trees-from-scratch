import numpy as np
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
        weights_exponent=None,
        handle_non_aranged_classes=False,
        random_state=None,
    ):
        self.criterion = criterion
        self.depth = depth
        self.max_depth = max_depth
        self.weights_exponent = weights_exponent
        self.handle_non_aranged_classes = handle_non_aranged_classes
        self.random_state = random_state

    def fit(self, X, y):
        # X, y = check_X_y(X, y)
        self.classes_ = unique_labels(y)

        if (
            self.weights_exponent is not None
            and self.criterion != "weighted_information_gain"
            and self.criterion != "qwk_weighted_information_gain"
            and self.criterion != "weighted_impurity"
        ):
            raise ValueError(
                "weighted_ig_power can only be used with criterion='weighted_information_gain'"
            )

        if not np.array_equal(self.classes_, np.arange(max(self.classes_) + 1)):
            if not self.handle_non_aranged_classes:
                raise ValueError("Classes are not aranged from 0 to the highest class. Set handle_non_aranged_classes argument to True to avoid this error.")

        if self.criterion == "information_gain":
            self._criterion = criterias.InformationGain()
        elif self.criterion == "gini":
            self._criterion = criterias.GiniSimpson()
        elif self.criterion == "ordinal_gini":
            self._criterion = criterias.OrdinalGiniSimpson()
        elif self.criterion == "weighted_information_gain":
            self._criterion = (
                criterias.WeightedIG(power=self.weights_exponent)
                if self.weights_exponent is not None
                else criterias.WeightedIG()
            )
        elif self.criterion == "ranking_impurity":
            self._criterion = criterias.RankingImpurity()
        elif self.criterion == "weighted_impurity":
            self._criterion = criterias.WeightedImpurity(n_classes=max(self.classes_) + 1, power=self.weights_exponent)
        elif self.criterion == "nysia_impurity":
            self._criterion = criterias.NysiaImpurity()
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
