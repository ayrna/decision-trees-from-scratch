import os

import decision_trees_from_scratch
import numpy as np
import pandas as pd
from decision_trees_from_scratch.decision_tree_classifier import DTC
from sklearn.metrics import mean_absolute_error
from sklearn.tree import DecisionTreeClassifier
from sklearn.utils import compute_sample_weight


def test_with_sklearn():
    data = pd.read_csv(
        os.path.join(decision_trees_from_scratch.__path__[0], "tests/test_data_exercise_tracking.csv")
    )
    X = data.drop(columns=["y"])
    y = data["y"]

    my_dtc = DTC(criterion="gini", max_depth=2)
    sk_dtc = DecisionTreeClassifier(criterion="gini", max_depth=2)
    my_dtc.fit(X, y, sample_weight=None)
    sk_dtc.fit(X, y, sample_weight=None)
    my_dtc_pred = my_dtc.predict(X)
    sk_dtc_pred = sk_dtc.predict(X)
    my_mae = mean_absolute_error(y, my_dtc_pred)
    sk_mae = mean_absolute_error(y, sk_dtc_pred)

    assert np.isclose(my_mae, sk_mae, atol=1e-8, rtol=1e-8)

    sample_weight = compute_sample_weight("balanced", y)

    my_dtc = DTC(criterion="gini", max_depth=2)
    sk_dtc = DecisionTreeClassifier(criterion="gini", max_depth=2)
    my_dtc.fit(X, y, sample_weight=sample_weight)
    sk_dtc.fit(X, y, sample_weight=sample_weight)
    my_dtc_pred = my_dtc.predict(X)
    sk_dtc_pred = sk_dtc.predict(X)
    my_mae = mean_absolute_error(y, my_dtc_pred)
    sk_mae = mean_absolute_error(y, sk_dtc_pred)

    assert np.isclose(my_mae, sk_mae, atol=1e-8, rtol=1e-8)
