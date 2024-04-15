from dataset import load_train_test
import numpy as np

from sklearn.ensemble import RandomForestRegressor
from sklearn import metrics

import matplotlib.pyplot as plt

if __name__ == "__main__":
    num_features = [5, 25, 50, 100]

    # Range of `n_estimators` values to explore.
    min_estimators = 25
    max_estimators = 750

    for features in num_features:
        pca = features
        train_dataset, test_dataset = load_train_test("MICRO2D_homogenized.h5", pca)

        rf = RandomForestRegressor (
                warm_start=True,
                max_features=None,
                oob_score=True,
                random_state=2024
        )

        print("Start OOB Error Validation")
        error_rate = []
        for i in range(min_estimators, max_estimators + 1, 25):
            print(f"Training with n_estimators={i}")
            rf.set_params(n_estimators=i)
            rf.fit(train_dataset.structs, train_dataset.Ex)

            oob_error = 1 - rf.oob_score_
            error_rate.append((i, oob_error))

        plt.plot(*zip(*error_rate), label=f'{features} features')

        min_error_estimator, min_error = min(error_rate, key = lambda t: t[1])[0], min(error_rate, key = lambda t: t[1])[1]
        plt.axhline(y = min_error, color = 'r', linestyle = '--')
        print(f"\nMinimum OOB Error: {min_error} at {min_error_estimator} estimators\n")
        
        plt.xlim(min_estimators, max_estimators)
        plt.title("Random Forest Validation Using Out of Bag Error")
        plt.xlabel("n_estimators")
        plt.ylabel("OOB error rate")
    
        plt.savefig(f"rf_{features}_validate.png",format = "png",dpi=300,bbox_inches='tight')
        plt.clf()

        rf = RandomForestRegressor (
                n_estimators=min_error_estimator,
                max_features=None,
                random_state=2024
        )

        rf.fit(train_dataset.structs, train_dataset.Ex)

        y_pred = rf.predict(test_dataset.structs)

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

        plt.savefig(f"rf_{features}_parity.png",format = "png",dpi=300,bbox_inches='tight')
        plt.clf()

        print('Mean Absolute Percentage Error (MAPE):', metrics.mean_absolute_percentage_error(test_dataset.Ex, y_pred))