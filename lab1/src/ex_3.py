from scipy.stats import zscore
from pyod.utils.data import generate_data
from sklearn.metrics import balanced_accuracy_score
import numpy as np

X_train, X_test, y_train, y_test = generate_data(
    contamination=0.1,
    n_train=1000,
    n_test=0,
    n_features=1
)

z_scores = np.abs(zscore(X_train))
threshold = np.quantile(z_scores, 0.9)

y_pred = [int(val >= threshold) for row in z_scores for val in row]

ba = balanced_accuracy_score(y_train, y_pred)
print(f"ba: {ba}")
