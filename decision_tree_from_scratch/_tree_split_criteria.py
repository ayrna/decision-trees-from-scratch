import math
import numpy as np
from decision_tree_from_scratch._tree_split_aux import ClassDistribution


### Information Gain.
##
#
class InformationGain:
    def __init__(self, n_classes):
        self.n_classes = n_classes

    def _entropy_single(self, y):
        cd = ClassDistribution(y, self.n_classes)
        entropy = -np.sum(cd.get_non_zero_probas() * np.log2(cd.get_non_zero_probas()))
        return entropy

    def compute(self, y, left_y, right_y, **kwargs):
        n_father = len(y)
        n_left = len(left_y)
        n_right = len(right_y)

        entropy_father = self._entropy_single(y)
        entropy_left = self._entropy_single(left_y)
        entropy_right = self._entropy_single(right_y)

        split_entropy = ((n_left / n_father) * entropy_left) + ((n_right / n_father) * entropy_right)

        return entropy_father - split_entropy


### Gini-index
##
#
class Gini:
    def __init__(self, n_classes):
        self.n_classes = n_classes

    def _compute_single(self, y):
        cd = ClassDistribution(y, self.n_classes)
        gini = 1 - np.sum(cd.get_probas() ** 2)
        return gini

    def compute(self, y, left_y, right_y, **kwargs):
        cd_father = ClassDistribution(y, self.n_classes)
        cd_left = ClassDistribution(left_y, self.n_classes)
        cd_right = ClassDistribution(right_y, self.n_classes)

        gini_father = 1 - np.sum(cd_father.get_probas() ** 2)
        gini_left = 1 - np.sum(cd_left.get_probas() ** 2)
        gini_right = 1 - np.sum(cd_right.get_probas() ** 2)

        PL = len(left_y) / len(y)
        PR = len(right_y) / len(y)

        return gini_father - (PL * gini_left) - (PR * gini_right)


### Ordinal Gini-Simpson. (Raffaella Piccarreta. 2007.)
##
#
class OrdinalGini:
    def __init__(self, n_classes):
        self.n_classes = n_classes

    def _compute_single_ogini(self, y):
        cd = ClassDistribution(y, self.n_classes)
        cumprobas = cd.get_cumprobas()
        ogini = 0
        for i in range(len(cumprobas)):
            ogini += cumprobas[i] * (1 - cumprobas[i])
        return ogini

    def compute(self, y, left_y, right_y, **kwargs):
        ogini_father = self._compute_single_ogini(y)
        ogini_left = self._compute_single_ogini(left_y)
        ogini_right = self._compute_single_ogini(right_y)

        PL = len(left_y) / len(y)
        PR = len(right_y) / len(y)

        return ogini_father - (PL * ogini_left) - (PR * ogini_right)


### Weighted IG. (Fen Xia, Wensheng Zhang, and Jue Wang. 2006)
##
#
class WeightedInformationGain:
    def __init__(self, n_classes, power=2):
        self.n_classes = n_classes
        self.power = power

    def _weighted_entropy_single(self, y, weights):
        cd = ClassDistribution(y, self.n_classes)
        w = np.array([weights[u] for u in cd.get_non_zero_labels()])
        entropy = -np.sum(w * cd.get_non_zero_probas() * np.log2(cd.get_non_zero_probas()))
        return entropy

    def _get_weights(self, y, unique_classes, power):
        mode_class = np.argmax(np.bincount(y))

        weight_denominator = np.sum(np.power(np.abs(unique_classes - mode_class), power))
        weights = {
            u: (
                np.power(abs(u - mode_class), power) / weight_denominator
                if not math.isclose(weight_denominator, 0.0)
                else 0
            )
            for u in unique_classes
        }
        return weights

    def compute(self, y, left_y, right_y, **kwargs):
        unique_y = np.unique(y)

        weights_father = self._get_weights(y, unique_classes=unique_y, power=self.power)
        weights_left = self._get_weights(left_y, unique_classes=unique_y, power=self.power)
        weights_right = self._get_weights(right_y, unique_classes=unique_y, power=self.power)

        entropy_father = self._weighted_entropy_single(y, weights=weights_father)
        entropy_left = self._weighted_entropy_single(left_y, weights=weights_left)
        entropy_right = self._weighted_entropy_single(right_y, weights=weights_right)

        n_father = len(y)
        n_left = len(left_y)
        n_right = len(right_y)

        split_entropy = ((n_left / n_father) * entropy_left) + ((n_right / n_father) * entropy_right)

        return entropy_father - split_entropy


### Ranking Impurity. (Fen Xia, Wensheng Zhang, and Jue Wang. 2006)
##
#
class RankingImpurity:
    def __init__(self, n_classes):
        self.n_classes = n_classes

    def _ranking_impurity_single(self, y):
        cd = ClassDistribution(y, self.n_classes)
        ri = 0
        for j in range(self.n_classes):
            for i in range(j):
                # Note that we do not need absolute value below, as labels[j] > labels[i] is satisfied for al i, j.
                ri += (cd.labels[j] - cd.labels[i]) * cd.counts[i] * cd.counts[j]
        return ri

    def compute(self, y, left_y, right_y, **kwargs):
        ri_father = self._ranking_impurity_single(y)
        ri_left = self._ranking_impurity_single(left_y)
        ri_right = self._ranking_impurity_single(right_y)

        return ri_father - ri_left - ri_right
