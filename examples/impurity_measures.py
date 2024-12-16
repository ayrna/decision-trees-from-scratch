import numpy as np
from decision_tree_from_scratch._tree_split_criteria import (
    Gini,
    InformationGain,
    OrdinalGini,
    WeightedInformationGain,
    RankingImpurity,
)

"""
Example computation of the impurity measures considered in [1].

The "y_upper" and "y_lower" arrays represent the upper and lower distributions of figure 2 in [1].

.. [1] Ayllón-Gavilán, R., Martinez-Estudillo, F., Guijo-Rubio, D., Hervás-Martínez, C. & Gutiérrez, P. A. (2024).
   Splitting criteria for ordinal decision trees: an experimental study. TODO:(place arxiv url here).
"""

y_upper = np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3])
y_lower = np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1])

# Gini
g = Gini(n_classes=4)
g_A = g.node_impurity(y_upper)
g_B = g.node_impurity(y_lower)
print("Upper set Gini :", g_A)
print("Lower set Gini:", g_B)
print("")

# Entropy
g = InformationGain(n_classes=4)
g_A = g.node_impurity(y_upper)
g_B = g.node_impurity(y_lower)
print("Upper set Entropy :", g_A)
print("Lower set Entropy:", g_B)
print("")

# Ordinal gini
og = OrdinalGini(n_classes=4)
og_A = og.node_impurity(y_upper)
og_B = og.node_impurity(y_lower)
print("Upper set Ordinal Gini :", og_A)
print("Lower set Ordinal Gini:", og_B)
print("")

# Weighted Entropy
c = WeightedInformationGain(n_classes=4, power=1)
unique_y = np.arange(4)
weights = c._get_weights(y_upper, unique_classes=unique_y)
we_A = c.node_impurity(y_upper, weights=weights)
we_B = c.node_impurity(y_lower, weights=weights)
print("Upper set Weighted Entropy :", we_A)
print("Lower set Weighted Entropy:", we_B)
print("")

# Ranking Impurity
ri = RankingImpurity(n_classes=4)
ri_A = ri.node_impurity(y_upper)
ri_B = ri.node_impurity(y_lower)
print("Upper set Ranking Impurity :", ri_A)
print("Lower set Ranking Impurity:", ri_B)
print("")
