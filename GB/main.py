from dataset import load_train_test
import numpy as np

from sklearn.ensemble import GradientBoostingRegressor
from sklearn import metrics

import matplotlib.pyplot as plt

if __name__ == "__main__":
    train_dataset, test_dataset = load_train_test("MICRO2D_homogenized.h5")

    subsample=0.5
    gb = GradientBoostingRegressor (
            warm_start=True,
            max_features=None,
            subsample=subsample,
            random_state=2024
    )

    # Range of `n_estimators` values to explore.
    min_estimators = 1000
    max_estimators = 10000

    print(f"Using Subsample = {subsample}")

    print("Start OOB Score Validation")
    oob_scores = []
    for i in range(min_estimators, max_estimators + 1, 1000):
        print(f"Training with n_estimators={i}")
        gb.set_params(n_estimators=i)
        gb.fit(train_dataset.structs, train_dataset.Ex)

        oob_score = gb.oob_score_
        oob_scores.append((i, oob_score))

    plt.plot(*zip(*oob_scores))
    plt.xlim(min_estimators, max_estimators)
    plt.title("Gradient Boosted Trees Validation Using Out of Bag Score")
    plt.xlabel("n_estimators")
    plt.ylabel("OOB Score")
    plt.show()

    min_score_estimator, min_score = min(oob_scores, key = lambda t: t[1])[0], min(oob_scores, key = lambda t: t[1])[1]

    gb = GradientBoostingRegressor (
            n_estimators=min_score_estimator,
            max_features=None,
            subsample=subsample,
            random_state=2024
    )

    print(f"n_estimators = {min_score_estimator}")
    print(f"Lowest OOB Score = {min_score}")
    gb.fit(train_dataset.structs, train_dataset.Ex)

    y_pred = gb.predict(test_dataset.structs)

    print('Mean Absolute Percentage Error (MAPE):', metrics.mean_absolute_percentage_error(test_dataset.Ex, y_pred))
