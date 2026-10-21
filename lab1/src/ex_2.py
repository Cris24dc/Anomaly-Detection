import matplotlib.pyplot as plt
from pyod.utils.data import generate_data
from pyod.models.knn import KNN
from sklearn.metrics import confusion_matrix, roc_curve, auc

X_train, X_test, y_train, y_test = generate_data(
    n_train=400,
    n_test=100,
    contamination=0.1
)

model = KNN(contamination=0.1)
model.fit(X_train)

y_train_pred = model.labels_
y_test_pred = model.predict(X_test)


tn, fp, fn, tp = confusion_matrix(y_test, y_test_pred).ravel()
tpr = tp/(tp+fn)
tnr = tn/(tn+fp)
ba = (tpr+tnr)/2
print(f"ba: {ba}")


y_test_scores = model.decision_function(X_test)
fpr, tpr, thresholds = roc_curve(y_test, y_test_scores)
roc_auc = auc(fpr, tpr)


plt.figure()
plt.plot(fpr, tpr, color='coral', lw=2, label=f'ROC Curve (AUC = {roc_auc:.3f})')
plt.title('ROC Curve')
plt.xlabel('FPR')
plt.ylabel('TPR')
plt.legend(loc="lower right")
plt.savefig('./img/Ex_2.pdf', format='pdf')

# daca contamination este modificat intr-o valoare mai mica,
# modelul marcheaza putini outliers si apar FN. In schimb daca
# este modificat intr-o valoare mai mare, pragul devine permisiv
# si apar FP
