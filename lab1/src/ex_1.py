from pyod.utils.data import generate_data
import matplotlib.pyplot as plt

X_train, X_test, y_train, y_test = generate_data(
    n_train=400,
    n_test=100,
    contamination=0.1
)

plt.figure()
inliers = X_train[y_train == 0, 0], X_train[y_train == 0, 1]
outliers = X_train[y_train == 1, 0], X_train[y_train == 1, 1]
plt.scatter(inliers[0], inliers[1], c='yellowgreen', label='Inliers')
plt.scatter(outliers[0], outliers[1], c='coral', label='Outliers')

plt.title('Training samples')
plt.xlabel('Feat1')
plt.ylabel('Feat2')
plt.legend()
plt.savefig('./img/Ex_1.pdf', format='pdf')
