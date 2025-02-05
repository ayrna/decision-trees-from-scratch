import numpy as np
from decision_trees_from_scratch._tree_split_aux import ClassDistribution
from decision_trees_from_scratch._tree_split_criteria_base import SplitCriterion


class InformationGain(SplitCriterion):
    """
    Information Gain (IG) split criterion.

    The Information Gain criterion is a measure of the reduction in entropy that results from splitting a node.
    """

    def _node_impurity(self, y, sample_weight):
        cd = ClassDistribution(y, self.n_classes, sample_weight=sample_weight)
        entropy = -np.sum(cd.get_non_zero_probas() * np.log2(cd.get_non_zero_probas()))
        return entropy

    def _compute(self, y, y_left, y_right, sw, sw_left, sw_right, **kwargs):
        entropy_parent = self.node_impurity(y, sample_weight=sw)
        entropy_left = self.node_impurity(y_left, sample_weight=sw_left)
        entropy_right = self.node_impurity(y_right, sample_weight=sw_right)

        weight_node_left, weight_node_right = self._compute_node_weights(y, y_left, y_right, sw, sw_left, sw_right)

        split_entropy = (weight_node_left * entropy_left) + (weight_node_right * entropy_right)
        return (entropy_parent - split_entropy, entropy_parent)


class Gini(SplitCriterion):
    """
    Gini split criterion.

    The Gini criterion computes the decrease in gini-index impurity. This impurity measure is the sum of the
    squared probabilities of each class.
    """

    def _node_impurity(self, y, sample_weight):
        cd = ClassDistribution(y, self.n_classes, sample_weight=sample_weight)
        gini = 1 - np.sum(cd.get_probas() ** 2)
        return gini

    def _compute(self, y, y_left, y_right, sw, sw_left, sw_right, **kwargs):
        cd_parent = ClassDistribution(y, self.n_classes, sample_weight=sw)
        cd_left = ClassDistribution(y_left, self.n_classes, sample_weight=sw_left)
        cd_right = ClassDistribution(y_right, self.n_classes, sample_weight=sw_right)

        gini_parent = 1 - np.sum(cd_parent.get_probas() ** 2)
        gini_left = 1 - np.sum(cd_left.get_probas() ** 2)
        gini_right = 1 - np.sum(cd_right.get_probas() ** 2)

        weight_node_left, weight_node_right = self._compute_node_weights(y, y_left, y_right, sw, sw_left, sw_right)

        return (gini_parent - (weight_node_left * gini_left) - (weight_node_right * gini_right), gini_parent)


class OrdinalGini(SplitCriterion):
    """
    Ordinal Gini (OGini) split criterion [1].

    The OGini criterion computes the decrease in ordinal gini impurity. This impurity measure is
    the sum of the squared cumulative probabilities of each class.

    ····
    [1] Raffaella Piccarreta. 2007. A new impurity measure for classification trees based on the Gini index.
    """

    def _node_impurity(self, y, sample_weight):
        cd = ClassDistribution(y, self.n_classes, sample_weight=sample_weight)
        cumprobas = cd.get_cumprobas()
        ogini = 0
        for i in range(len(cumprobas)):
            ogini += cumprobas[i] * (1 - cumprobas[i])
        return ogini

    def _compute(self, y, y_left, y_right, sw, sw_left, sw_right, **kwargs):
        ogini_parent = self.node_impurity(y, sample_weight=sw)
        ogini_left = self.node_impurity(y_left, sample_weight=sw_left)
        ogini_right = self.node_impurity(y_right, sample_weight=sw_right)

        weight_node_left, weight_node_right = self._compute_node_weights(y, y_left, y_right, sw, sw_left, sw_right)

        return (ogini_parent - (weight_node_left * ogini_left) - (weight_node_right * ogini_right), ogini_parent)


class WeightedInformationGain(SplitCriterion):
    """
    Weighted Information Gain (WIG) split criterion [1].

    The WIG criterion computes the decrease in weighted entropy that results from splitting a node. The impurity of
    a node is calculated as the weighted entropy of the node.

    ····
    [1] Singer, G., Anuar, R., & Ben-Gal, I. (2020). A weighted information-gain measure for ordinal classification
      trees. Expert Systems with Applications, 152, 113375.
    """

    def __init__(self, n_classes, power=1):
        """
        Parameters:
        -----------
        n_classes (int): Number of classes in the target variable.
        power (int): Power of the weights. Default is 1.
        """
        super().__init__(n_classes)
        self.power = power

    def _node_impurity(self, y, weights, sample_weight):
        cd = ClassDistribution(y, self.n_classes, sample_weight=sample_weight)
        w = np.array([weights[u] for u in cd.get_non_zero_labels()])
        entropy = -np.sum(w * cd.get_non_zero_probas() * np.log2(cd.get_non_zero_probas()))
        return entropy

    def _get_weights(self, y, unique_classes):
        mode_class = np.argmax(np.bincount(y))

        weight_denominator = np.sum(np.power(np.abs(unique_classes - mode_class), self.power))
        weights = {
            u: (
                np.power(abs(u - mode_class), self.power) / weight_denominator
                if not np.isclose(weight_denominator, 0.0)
                else 0
            )
            for u in unique_classes
        }
        return weights

    def _compute(self, y, y_left, y_right, sw, sw_left, sw_right, **kwargs):
        unique_y = np.unique(y)

        weights_parent = self._get_weights(y, unique_classes=unique_y)
        weights_left = self._get_weights(y_left, unique_classes=unique_y)
        weights_right = self._get_weights(y_right, unique_classes=unique_y)

        entropy_parent = self.node_impurity(y, weights=weights_parent, sample_weight=sw)
        entropy_left = self.node_impurity(y_left, weights=weights_left, sample_weight=sw_left)
        entropy_right = self.node_impurity(y_right, weights=weights_right, sample_weight=sw_right)

        weight_node_left, weight_node_right = self._compute_node_weights(y, y_left, y_right, sw, sw_left, sw_right)

        split_entropy = (weight_node_left * entropy_left) + (weight_node_right * entropy_right)

        return (entropy_parent - split_entropy, entropy_parent)


class RankingImpurity(SplitCriterion):
    """
    Ranking Impurity (RI) split criterion [1].

    The RI criterion measures the decrease in RI that results from the split. The RI is the potential maximum
    number of missclassified samples based on a given node distribution.

    ····
    [1] Xia, F., Zhang, W., & Wang, J. (2006). An Effective Tree-Based Algorithm for Ordinal Regression.
    IEEE Intell. Informatics Bull., 7(1), 22-26.
    """

    def _node_impurity(self, y, sample_weight):
        cd = ClassDistribution(y, self.n_classes, sample_weight=sample_weight)
        ri = 0
        for j in range(self.n_classes):
            for i in range(j):
                # Note that we do not need absolute value below, as labels[j] > labels[i] is satisfied for al i, j.
                ri += (cd.labels[j] - cd.labels[i]) * cd.counts[i] * cd.counts[j]
        return ri

    def _compute(self, y, y_left, y_right, sw, sw_left, sw_right, **kwargs):
        ri_parent = self.node_impurity(y, sample_weight=sw)
        ri_left = self.node_impurity(y_left, sample_weight=sw_left)
        ri_right = self.node_impurity(y_right, sample_weight=sw_right)

        return (ri_parent - ri_left - ri_right, ri_parent)
