import numpy as np
from decision_tree_from_scratch._tree_split_criteria import (
    Gini,
    InformationGain,
    OrdinalGini,
    WeightedInformationGain,
    RankingImpurity,
)

"""
Example computation of the split criteria considered in [1].

The "left" and "right" splits corresponds to the example presented in Figure 3 in [1].

.. [1] Ayllón-Gavilán, R., Martinez-Estudillo, F., Guijo-Rubio, D., Hervás-Martínez, C. & Gutiérrez, P. A. (2024).
   Splitting criteria for ordinal decision trees: an experimental study. TODO:(place arxiv url here).
"""

# fmt: off
y_father = np.array([0 ,0 ,0 ,0 ,0 ,0 ,0 ,0 ,0 ,0 ,1 ,1 ,1 ,1 ,1 ,1 ,1 ,1 ,1 ,1 ,2 ,2 ,2 ,2 ,2 ,2 ,2 ,2 ,2 ,2 ,3 ,3 ,3 ,3 ,3 ,3 ,3 ,3 ,3 ,3])
# fmt: on

left_y_left = np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3])
left_y_right = np.array([1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2])
right_y_left = np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1])
right_y_right = np.array([2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3])

gini = Gini(n_classes=4)
ig = InformationGain(n_classes=4)
ogini = OrdinalGini(n_classes=4)
wig = WeightedInformationGain(n_classes=4, power=1)
ri = RankingImpurity(n_classes=4)

print("Left Split Gini =", gini.compute(y_father, left_y_left, left_y_right))
print("Left Split IG =", ig.compute(y_father, left_y_left, left_y_right))
print("Left Split OG =", ogini.compute(y_father, left_y_left, left_y_right))
print("Left Split WIG =", wig.compute(y_father, left_y_left, left_y_right))
print("Left Split RI =", ri.compute(y_father, left_y_left, left_y_right))
print("")
print("Right Split Gini =", gini.compute(y_father, right_y_left, right_y_right))
print("Right Split IG =", ig.compute(y_father, right_y_left, right_y_right))
print("Right Split OG =", ogini.compute(y_father, right_y_left, right_y_right))
print("Right Split WIG =", wig.compute(y_father, right_y_left, right_y_right))
print("Right Split RI =", ri.compute(y_father, right_y_left, right_y_right))
