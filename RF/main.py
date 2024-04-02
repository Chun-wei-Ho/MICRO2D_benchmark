from dataset import load_train_test
import numpy as np

from sklearn.ensemble import RandomForestRegressor
from sklearn import metrics

import matplotlib.pyplot as plt

if __name__ == "__main__":
    train_dataset, test_dataset = load_train_test("MICRO2D_homogenized.h5")

    rf = RandomForestRegressor (
            warm_start=True,
            max_features=None,
            oob_score=True,
            random_state=2024
    )

    # Range of `n_estimators` values to explore.
    min_estimators = 50
    max_estimators = 500

    print("Validating")
    error_rate = []
    for i in range(min_estimators, max_estimators + 1, 10):
        rf.set_params(n_estimators=i)
        rf.fit(train_dataset.structs, train_dataset.Ex)

        oob_error = 1 - rf.oob_score_
        error_rate.append((i, oob_error))

    plt.plot(*zip(*error_rate))
    plt.xlim(min_estimators, max_estimators)
    plt.xlabel("n_estimators")
    plt.ylabel("OOB error rate")
    plt.show()

    min_error_estimator, min_error = min(error_rate, key = lambda t: t[1])[0], min(error_rate, key = lambda t: t[1])[1]

    rf = RandomForestRegressor (
            n_estimators=min_error_estimator,
            max_features=None,
            random_state=2024
    )

    print(f"n_estimators = {min_error_estimator}")
    print(f"Lowest OOB error = {min_error}")
    rf.fit(train_dataset.structs, train_dataset.Ex)

    y_pred = rf.predict(test_dataset.structs)

    mse = metrics.mean_squared_error(test_dataset.Ex, y_pred)
    print('Mean Absolute Error (MAE):', metrics.mean_absolute_error(test_dataset.Ex, y_pred))
