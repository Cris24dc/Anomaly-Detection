from scipy.stats import zscore
from sklearn.metrics import balanced_accuracy_score
import numpy as np

n_samples = 1000
n_outliers = int(n_samples * 0.1)
n_inliers = n_samples - n_outliers


A = np.random.randn(3, 3)
mu = np.random.uniform(-5, 5, 3)
sigma = A @ A.T + 0.1 * np.eye(3)
L = np.linalg.cholesky(sigma)


x_in = np.random.randn(n_inliers, 3)
x_out = 5 * np.random.randn(n_outliers, 3)  
y_in = x_in @ L.T + mu
y_out = x_out @ L.T + mu

X = np.vstack((y_in, y_out))
y_true = np.concatenate((np.zeros(n_inliers), np.ones(n_outliers)))


z_scores = np.abs(zscore(X, axis=0))
scores = np.linalg.norm(z_scores, axis=1)
threshold = np.quantile(scores, 0.9)
y_pred = (scores >= threshold).astype(int)

ba = balanced_accuracy_score(y_true, y_pred)
print(f"ba: {ba}")
