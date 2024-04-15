from dataset import load_train_test
import numpy as np

from sklearn.ensemble import GradientBoostingRegressor
from sklearn import metrics

import matplotlib.pyplot as plt

if __name__ == "__main__":
    num_features = [5, 25, 50, 100]

    subsample=0.5
    print(f"Using Subsample = {subsample}")

    # Range of `n_estimators` values to explore.
    min_estimators = 1000
    max_estimators = 10000

    for features in num_features:
        pca = features
        train_dataset, test_dataset = load_train_test("MICRO2D_homogenized.h5", pca)

        gb = GradientBoostingRegressor (
                warm_start=True,
                max_features=None,
                subsample=subsample,
                random_state=2024
        )

        print("Start OOB Score Validation")
        oob_scores = []

        for i in range(min_estimators, max_estimators + 1, 1000):
            print(f"Training with n_estimators={i}")
            gb.set_params(n_estimators=i)
            gb.fit(train_dataset.structs, train_dataset.Ex)

            oob_score = gb.oob_score_
            oob_scores.append((i, oob_score))

        min_score_estimator, min_score = min(oob_scores, key = lambda t: t[1])[0], min(oob_scores, key = lambda t: t[1])[1]

        plt.plot(*zip(*oob_scores))
        plt.axhline(y = min_score, color = 'r', linestyle = '--')
        print(f"\nMinimum OOB Score: {min_score} at {min_score_estimator} estimators\n")

        plt.xlim(min_estimators, max_estimators)
        plt.title("Gradient Boosted Trees Validation Using Out of Bag Score")
        plt.xlabel("n_estimators")
        plt.ylabel("OOB Score")

        plt.savefig(f"gb_{features}_validate.png",format = "png",dpi=300,bbox_inches='tight')
        plt.clf()

        gb = GradientBoostingRegressor (
                n_estimators=min_score_estimator,
                max_features=None,
                subsample=subsample,
                random_state=2024
        )

        gb.fit(train_dataset.structs, train_dataset.Ex)

        y_pred = gb.predict(test_dataset.structs)

        # Plot parity plot
        x = test_dataset.Ex
        y = y_pred

        bounds = (min(x.min(), y.min()) - int(0.1 * y.min()), max(x.max(), y.max())+ int(0.1 * y.max()))
        ax = plt.gca()
        ax.set_xlim(bounds)
        ax.set_ylim(bounds)

        ax.plot([0, 1], [0, 1], "r-",lw=2 ,transform=ax.transAxes)

        plt.plot(x, y, "o")
        plt.title("Random Forest Parity Plot")
        plt.xlabel('Ground Truth')
        plt.ylabel('Prediction')

        plt.savefig(f"gb_{features}_parity.png",format = "png",dpi=300,bbox_inches='tight')
        plt.clf()

        print('Mean Absolute Percentage Error (MAPE):', metrics.mean_absolute_percentage_error(test_dataset.Ex, y_pred))
