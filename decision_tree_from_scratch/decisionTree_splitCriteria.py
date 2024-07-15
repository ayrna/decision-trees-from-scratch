import math
import numpy as np


### Gini-Simpson. (Raffaella Piccarreta. 2007.)
##
#
class GiniSimpson:
    def __init__(self):
        pass

    def compute(self, target, left_target, right_target):
        labels, counts = np.unique(target, return_counts=True)
        labels_L, counts_L = np.unique(left_target, return_counts=True)
        labels_R, counts_R = np.unique(right_target, return_counts=True)

        f = {l: c for l, c in zip(labels, counts)}
        f_L = {l: c for l, c in zip(labels_L, counts_L)}
        f_R = {l: c for l, c in zip(labels_R, counts_R)}

        gini_father = 0
        gini_left = 0
        gini_right = 0
        for l, c in f.items():
            gini_father += (c / len(target)) ** 2
            gini_left += (f_L[l] / len(left_target)) ** 2 if l in labels_L else 0
            gini_right += (f_R[l] / len(right_target)) ** 2 if l in labels_R else 0
        gini_father = 1 - gini_father
        gini_left = 1 - gini_left
        gini_right = 1 - gini_right

        PL = len(left_target) / len(target)
        PR = len(right_target) / len(target)

        return gini_father - (PL * gini_left) - (PR * gini_right)


### Ordinal Gini-Simpson. (Raffaella Piccarreta. 2007.)
##
#
class OrdinalGiniSimpson:
    def __init__(self):
        pass

    def compute(self, target, left_target, right_target):
        labels, counts = np.unique(target, return_counts=True)
        labels_L, counts_L = np.unique(left_target, return_counts=True)
        labels_R, counts_R = np.unique(right_target, return_counts=True)

        f = {l: c for l, c in zip(labels, counts)}
        f_L = {l: c for l, c in zip(labels_L, counts_L)}
        f_R = {l: c for l, c in zip(labels_R, counts_R)}

        for l in f:
            if l not in f_L:
                f_L[l] = 0
            if l not in f_R:
                f_R[l] = 0

        p_L = np.cumsum(np.array(sorted(f_L.items()))[:, 1] / len(left_target))
        p_R = np.cumsum(np.array(sorted(f_R.items()))[:, 1] / len(right_target))

        PL = len(left_target) / len(target)
        PR = len(right_target) / len(target)

        gini = 0
        for p_L_i, p_R_i in zip(p_L, p_R):
            gini += np.power(p_L_i - p_R_i, 2)

        return PL * PR * gini


### Weighted IG. (Fen Xia, Wensheng Zhang, and Jue Wang. 2006)
##
#
class WeightedIG:
    def __init__(self, power=2):
        self.power = power

    def _weighted_entropy_single(self, y, weights):
        uni, class_counts = np.unique(y, return_counts=True)
        class_probabilities = class_counts / len(y)
        w = np.array([weights[u] for u in uni])
        entropy = -np.sum(w * class_probabilities * np.log2(class_probabilities))
        return entropy

    def _get_weights(self, y, unique_classes, power):
        mode_class = np.argmax(np.bincount(y))
        weight_denominator = np.sum(
            np.power(np.abs(unique_classes - mode_class), power)
        )
        weights = {
            u: (
                np.power(abs(u - mode_class), power) / weight_denominator
                if not math.isclose(weight_denominator, 0.0)
                else 0
            )
            for u in unique_classes
        }
        return weights

    def compute(self, target, left_target, right_target):
        unique_target = np.unique(target)

        weights_father = self._get_weights(
            target, unique_classes=unique_target, power=self.power
        )
        weights_left = self._get_weights(
            left_target, unique_classes=unique_target, power=self.power
        )
        weights_right = self._get_weights(
            right_target, unique_classes=unique_target, power=self.power
        )

        entropy_father = self._weighted_entropy_single(target, weights=weights_father)
        entropy_left = self._weighted_entropy_single(left_target, weights=weights_left)
        entropy_right = self._weighted_entropy_single(
            right_target, weights=weights_right
        )

        n_father = len(target)
        n_left = len(left_target)
        n_right = len(right_target)

        split_entropy = ((n_left / n_father) * entropy_left) + (
            (n_right / n_father) * entropy_right
        )

        return entropy_father - split_entropy


### Ranking Impurity. (Fen Xia, Wensheng Zhang, and Jue Wang. 2006)
##
#
class RankingImpurity:
    def __init__(self):
        pass

    def _ranking_impurity_single(self, target):
        labels, counts = np.unique(target, return_counts=True)
        ri = 0
        for j in range(len(labels)):
            for i in range(j):
                ri += (labels[j] - labels[i]) * counts[i] * counts[j]
        return ri

    def compute(self, target, left_target, right_target):
        ri_father = self._ranking_impurity_single(target)
        ri_left = self._ranking_impurity_single(left_target)
        ri_right = self._ranking_impurity_single(right_target)

        return ri_father - ri_left - ri_right


### Weighted Impurity.
##
#
def make_cost_matrix(num_ratings, power):
    cost_matrix = np.reshape(
        np.tile(range(num_ratings), num_ratings), (num_ratings, num_ratings)
    )
    cost_matrix = (
        np.power(cost_matrix - np.transpose(cost_matrix), power)
        / (num_ratings - 1) ** power
    )
    return np.float32(cost_matrix)


class WeightedImpurity:
    def __init__(self, n_classes, power):
        self.n_classes = n_classes
        self.power = power
        self.weights_matrix = make_cost_matrix(n_classes, self.power)

    def _weighted_impurity_single(self, target):
        labels, counts = np.unique(target, return_counts=True)
        ri = 0
        for j in range(len(labels)):
            for i in range(j):
                ri += self.weights_matrix[labels[i], labels[j]] * counts[i] * counts[j]
        return ri

    def compute(self, target, left_target, right_target):
        ri_father = self._weighted_impurity_single(target)
        ri_left = self._weighted_impurity_single(left_target)
        ri_right = self._weighted_impurity_single(right_target)

        return ri_father - ri_left - ri_right


### Nysia Weighted Impurity (Nysia I. George, Tzu-Pin Lu, and Ching-Wei Chang, 2016).
##
#
# class NysiaImpurity:
#     def __init__(self):
#         pass

#     def _weighted_impurity_single(self, target):
#         labels, counts = np.unique(target, return_counts=True)
#         ri = 0
#         for j in range(len(labels)):
#             for i in range(j):
#                 ri += ((len(target) - counts[i]) / counts[j]) * abs(
#                     labels[i] - labels[j]
#                 )
#         return ri

#     def compute(self, target, left_target, right_target):
#         ri_father = self._weighted_impurity_single(target)
#         ri_left = self._weighted_impurity_single(left_target)
#         ri_right = self._weighted_impurity_single(right_target)
#         return ri_father - (ri_left + ri_right)


# ### Information Gain.
# ##
# #
# class InformationGain:
#     def __init__(self):
#         pass

#     def _entropy_single(self, y):
#         _, class_counts = np.unique(y, return_counts=True)
#         class_probabilities = class_counts / len(y)
#         entropy = -np.sum(class_probabilities * np.log2(class_probabilities))
#         return entropy  # / len(class_probabilities)  # / n_classes

#     def compute(self, target, left_target, right_target):
#         n_father = len(target)
#         n_left = len(left_target)
#         n_right = len(right_target)

#         entropy_father = self._entropy_single(target)
#         entropy_left = self._entropy_single(left_target)
#         entropy_right = self._entropy_single(right_target)

#         split_entropy = ((n_left / n_father) * entropy_left) + (
#             (n_right / n_father) * entropy_right
#         )

#         return entropy_father - split_entropy


# ### Twoing Criterion. (Raffaella Piccarreta. 2007.)
# ##
# #
# class TwoingCriterion:
#     def __init__(self):
#         pass

#     def generate_subsets(self, classes):
#         return list(
#             chain.from_iterable(
#                 combinations(classes, r) for r in range(len(classes) + 1)
#             )
#         )

#     def _indiv_twoing_impurity(self, t, subset):
#         classes, counts = np.unique(t, return_counts=True)
#         class_frequency = {c: cou / len(t) for c, cou in zip(classes, counts)}

#         pi_c1 = sum([class_frequency[c] for c in classes if c in subset])
#         pi_c1_complement = sum([class_frequency[c] for c in classes if c not in subset])

#         return 2 * pi_c1 * pi_c1_complement

#     def compute(self, target, left_target, right_target):
#         classes = np.unique(target)
#         pL = len(left_target) / len(target)
#         pR = len(right_target) / len(target)

#         subsets = self.generate_subsets(classes)
#         unique_subsets = []
#         for subset in subsets:
#             subset_C = tuple([c for c in classes if c not in subset])
#             if (subset not in unique_subsets) and (subset_C not in unique_subsets):
#                 unique_subsets.append(list(subset))

#         highest_impurity_decrease = 0
#         for subset in unique_subsets:
#             impurity = self._indiv_twoing_impurity(target, subset)
#             impurity_L = self._indiv_twoing_impurity(left_target, subset)
#             impurity_R = self._indiv_twoing_impurity(right_target, subset)

#             impurity_decrease = impurity - (pL * impurity_L) - (pR * impurity_R)
#             if impurity_decrease > highest_impurity_decrease:
#                 highest_impurity_decrease = impurity_decrease

#         return highest_impurity_decrease


# ### Ordinal Twoing Criterion. (Raffaella Piccarreta. 2007.)
# ##
# #
# class OrdinalTwoingCriterion:
#     def __init__(self):
#         pass

#     def generate_ordered_subsets(self, classes):
#         subsets = []
#         for i in range(len(classes)):
#             subsets.append(tuple(classes[:i]))
#         return subsets

#     def _indiv_twoing_impurity(self, t, subset):
#         classes, counts = np.unique(t, return_counts=True)
#         class_frequency = {c: cou / len(t) for c, cou in zip(classes, counts)}

#         pi_c1 = sum([class_frequency[c] for c in classes if c in subset])
#         pi_c1_complement = sum([class_frequency[c] for c in classes if c not in subset])

#         return 2 * pi_c1 * pi_c1_complement

#     def compute(self, target, left_target, right_target):
#         classes = np.unique(target)
#         pL = len(left_target) / len(target)
#         pR = len(right_target) / len(target)

#         subsets = self.generate_ordered_subsets(classes)
#         unique_subsets = []
#         for subset in subsets:
#             subset_C = tuple([c for c in classes if c not in subset])
#             if (subset not in unique_subsets) and (subset_C not in unique_subsets):
#                 unique_subsets.append(list(subset))

#         highest_impurity_decrease = 0
#         for subset in subsets:
#             impurity = self._indiv_twoing_impurity(target, subset)
#             impurity_L = self._indiv_twoing_impurity(left_target, subset)
#             impurity_R = self._indiv_twoing_impurity(right_target, subset)

#             impurity_decrease = impurity - (pL * impurity_L) - (pR * impurity_R)
#             if impurity_decrease > highest_impurity_decrease:
#                 highest_impurity_decrease = impurity_decrease

#         return highest_impurity_decrease


### IG weighted with weights proposed in QWK loss (Jordi de la Torre, Domenec Puig, Aida Valls. 2017)
##
#
# class QWKWeightedIG:
#     def __init__(self, power=2):
#         self.power = power

#     def _weighted_entropy_single(self, y, weights):
#         uni, class_counts = np.unique(y, return_counts=True)
#         cp = {}
#         for i in range(len(uni)):
#             cp[uni[i]] = class_counts[i] / len(y)
#         w = np.array([weights[u] for u in uni])
#         entropy = -np.sum(w * class_probabilities * np.log2(class_probabilities))
#         return entropy

#     def _make_cost_matrix(self, num_ratings):
#         cost_matrix = np.reshape(
#             np.tile(range(num_ratings), num_ratings), (num_ratings, num_ratings)
#         )
#         cost_matrix = (
#             np.power(cost_matrix - np.transpose(cost_matrix), self.power)
#             / (num_ratings - 1) ** self.power
#         )
#         return np.float32(cost_matrix)

#     def compute(self, target, left_target, right_target):

#         weights = self._make_cost_matrix(len(np.unique(target)))

#         n_father = len(target)
#         n_left = len(left_target)
#         n_right = len(right_target)

#         entropy_left = self._weighted_entropy_single(left_target, weights=weights)
#         entropy_right = self._weighted_entropy_single(right_target, weights=weights)

#         split_entropy = ((n_left / n_father) * entropy_left) + (
#             (n_right / n_father) * entropy_right
#         )

#         return -split_entropy


# ### Twoing Criterion. (Raffaella Piccarreta. 2007.)
# ##
# #
# def twoing_criterion(parent_classes, left_classes, right_classes):
#     n_parent = len(parent_classes)
#     n_left = len(left_classes)
#     n_right = len(right_classes)

#     p_L = n_left / n_parent
#     p_R = n_right / n_parent

#     def class_distribution(classes):
#         distribution = {}
#         for c in classes:
#             if c in distribution:
#                 distribution[c] += 1
#             else:
#                 distribution[c] = 1
#         for c in distribution:
#             distribution[c] /= len(classes)
#         return distribution

#     parent_dist = class_distribution(parent_classes)
#     left_dist = class_distribution(left_classes)
#     right_dist = class_distribution(right_classes)

#     twoing_value = 0
#     for c in parent_dist:
#         pi_t_C1_L = left_dist.get(c, 0)
#         pi_t_C1_R = right_dist.get(c, 0)
#         twoing_value += (pi_t_C1_L - pi_t_C1_R) ** 2

#     twoing_value *= 2 * p_L * p_R

#     return twoing_value


# ######################################################################
# ######################################################################

# # def twoing_criterion(target, left_target, right_target):
# #     n_parent = len(target)
# #     n_left = len(left_target)
# #     n_right = len(right_target)

# #     p_L = n_left / n_parent
# #     p_R = n_right / n_parent

# #     _, parent_counts = np.unique(target, return_counts=True)
# #     _, left_counts = np.unique(left_target, return_counts=True)
# #     _, right_counts = np.unique(right_target, return_counts=True)

# #     parent_dist = parent_counts / n_parent
# #     left_dist = left_counts / n_left
# #     right_dist = right_counts / n_right

# #     twoing_value = 0
# #     for c in parent_dist:
# #         pi_t_C1_L = left_dist.get(c, 0)
# #         pi_t_C1_R = right_dist.get(c, 0)
# #         twoing_value += (pi_t_C1_L - pi_t_C1_R) ** 2

# #     twoing_value *= 2 * p_L * p_R

# #     return twoing_value


# # GINI ################################################################################
# #######################################################################################


# def efficient_gini_index(target, left_target, right_target):
#     total_samples = len(target)
#     left_samples = len(left_target)
#     right_samples = len(right_target)

#     if total_samples == 0:
#         return 0

#     _, left_counts = np.unique(left_target, return_counts=True)
#     left_probs = left_counts / left_samples
#     _, right_counts = np.unique(right_target, return_counts=True)
#     right_probs = right_counts / right_samples

#     left_gini = 1 - np.sum(left_probs**2)
#     right_gini = 1 - np.sum(right_probs**2)

#     weighted_gini = (left_samples / total_samples) * left_gini + (
#         right_samples / total_samples
#     ) * right_gini

#     return weighted_gini


# def gini(target, left_target, right_target):

#     _, left_target_counts = np.unique(left_target, return_counts=True)
#     _, right_target_counts = np.unique(right_target, return_counts=True)

#     GI = 0
#     n = len(target)

#     n_left = len(left_target)
#     left_counts = 0
#     for n_i in left_target_counts:
#         left_counts += (n_i / n_left) ** 2
#     GI += (n_left / n) * (1 - left_counts)

#     n_right = len(right_target)
#     right_counts = 0
#     for n_i in right_target_counts:
#         right_counts += (n_i / n_right) ** 2
#     GI += (n_right / n) * (1 - right_counts)

#     return GI
