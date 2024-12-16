import math
import numpy as np
from decision_tree_from_scratch._tree_split_aux import ClassDistribution
from decision_tree_from_scratch._tree_split_criteria_base import SplitCriterion


class InformationGain(SplitCriterion):
    """
    Information Gain (IG) split criterion.

    The Information Gain criterion is a measure of the reduction in entropy that results from splitting a node.
    """

    def _node_impurity(self, y):
        cd = ClassDistribution(y, self.n_classes)
        entropy = -np.sum(cd.get_non_zero_probas() * np.log2(cd.get_non_zero_probas()))
        return entropy

    def _compute(self, y, left_y, right_y, **kwargs):
        n_parent = len(y)
        n_left = len(left_y)
        n_right = len(right_y)

        entropy_parent = self.node_impurity(y)
        entropy_left = self.node_impurity(left_y)
        entropy_right = self.node_impurity(right_y)

        split_entropy = ((n_left / n_parent) * entropy_left) + ((n_right / n_parent) * entropy_right)

        return entropy_parent - split_entropy


class Gini(SplitCriterion):
    """
    Gini split criterion.

    The Gini criterion computes the decrease in gini-index impurity. This impurity measure is the sum of the
    squared probabilities of each class.
    """

    def _node_impurity(self, y):
        cd = ClassDistribution(y, self.n_classes)
        gini = 1 - np.sum(cd.get_probas() ** 2)
        return gini

    def _compute(self, y, left_y, right_y, **kwargs):
        cd_parent = ClassDistribution(y, self.n_classes)
        cd_left = ClassDistribution(left_y, self.n_classes)
        cd_right = ClassDistribution(right_y, self.n_classes)

        gini_parent = 1 - np.sum(cd_parent.get_probas() ** 2)
        gini_left = 1 - np.sum(cd_left.get_probas() ** 2)
        gini_right = 1 - np.sum(cd_right.get_probas() ** 2)

        PL = len(left_y) / len(y)
        PR = len(right_y) / len(y)

        return gini_parent - (PL * gini_left) - (PR * gini_right)


class OrdinalGini(SplitCriterion):
    """
    Ordinal Gini (OGini) split criterion [1].

    The OGini criterion computes the decrease in ordinal gini impurity. This impurity measure is
    the sum of the squared cumulative probabilities of each class.

    ····
    [1] Raffaella Piccarreta. 2007. A new impurity measure for classification trees based on the Gini index.
    """

    def _node_impurity(self, y):
        cd = ClassDistribution(y, self.n_classes)
        cumprobas = cd.get_cumprobas()
        ogini = 0
        for i in range(len(cumprobas)):
            ogini += cumprobas[i] * (1 - cumprobas[i])
        return ogini

    def _compute(self, y, left_y, right_y, **kwargs):
        ogini_parent = self.node_impurity(y)
        ogini_left = self.node_impurity(left_y)
        ogini_right = self.node_impurity(right_y)

        PL = len(left_y) / len(y)
        PR = len(right_y) / len(y)

        return ogini_parent - (PL * ogini_left) - (PR * ogini_right)


class WeightedInformationGain(SplitCriterion):
    """
    Weighted Information Gain (WIG) split criterion [1].

    The WIG criterion computes the decrease in weighted entropy that results from splitting a node. The impurity of
    a node is calculated as the weighted entropy of the node.

    ····
    [1] Singer, G., Anuar, R., & Ben-Gal, I. (2020). A weighted information-gain measure for ordinal classification
      trees. Expert Systems with Applications, 152, 113375.
    """

    def _node_impurity(self, y, weights):
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

    def _compute(self, y, left_y, right_y, **kwargs):
        unique_y = np.unique(y)

        weights_parent = self._get_weights(y, unique_classes=unique_y, power=self.power)
        weights_left = self._get_weights(left_y, unique_classes=unique_y, power=self.power)
        weights_right = self._get_weights(right_y, unique_classes=unique_y, power=self.power)

        entropy_parent = self.node_impurity(y, weights=weights_parent)
        entropy_left = self.node_impurity(left_y, weights=weights_left)
        entropy_right = self.node_impurity(right_y, weights=weights_right)

        n_parent = len(y)
        n_left = len(left_y)
        n_right = len(right_y)

        split_entropy = ((n_left / n_parent) * entropy_left) + ((n_right / n_parent) * entropy_right)

        return entropy_parent - split_entropy


class RankingImpurity(SplitCriterion):
    """
    Ranking Impurity (RI) split criterion [1].

    The RI criterion measures the decrease in RI that results from the split. The RI is the potential maximum
    number of missclassified samples based on a given node distribution.

    ····
    [1] Xia, F., Zhang, W., & Wang, J. (2006). An Effective Tree-Based Algorithm for Ordinal Regression.
    IEEE Intell. Informatics Bull., 7(1), 22-26.
    """

    def _node_impurity(self, y):
        cd = ClassDistribution(y, self.n_classes)
        ri = 0
        for j in range(self.n_classes):
            for i in range(j):
                # Note that we do not need absolute value below, as labels[j] > labels[i] is satisfied for al i, j.
                ri += (cd.labels[j] - cd.labels[i]) * cd.counts[i] * cd.counts[j]
        return ri

    def _compute(self, y, left_y, right_y, **kwargs):
        ri_parent = self.node_impurity(y)
        ri_left = self.node_impurity(left_y)
        ri_right = self.node_impurity(right_y)

        return ri_parent - ri_left - ri_right
