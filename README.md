This repository provides a from scratch sklearn-based implementation of the CART algorithm for classification. Our implementation introduces notable differences compared to the existing sklearn [DecisionTreeClassifier](https://scikit-learn.org/stable/modules/generated/sklearn.tree.DecisionTreeClassifier.html "DecisionTreeClassifier"):

* :rocket: It is **fully developed in python**. This enables researchers to easily tweak the algorithm and experiment with it for research purposes.

* :rocket: Includes current **state-of-the-art splitting criteria for ordinal regression** tasks, such as Ordinal Gini or Weighted Entropy.

:sparkles: Despite being developed from scratch, our implementation achieves **same accuracy results as scikit-learn implementation**.
