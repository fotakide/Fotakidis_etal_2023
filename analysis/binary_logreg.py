import pandas as pd
import numpy as np
import seaborn as sns

import statsmodels.api as sm
from statsmodels.formula.api import logit

from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.metrics import roc_curve, roc_auc_score, auc, accuracy_score

import matplotlib.pyplot as plt

plt.rc("font", size=14)

sns.set(style="white")
sns.set(style="whitegrid", color_codes=True)

data = pd.read_csv(
    "E:/Publications/BFAST_Monitor/results/blr/zonal_statistics/dnbr/all_dnbr_raslayer_zonal_statistics.csv",
    usecols=['zone', 'min'], low_memory=False)
data["zone"] = pd.to_numeric(data["zone"], errors='coerce')
data["min"] = pd.to_numeric(data["min"], errors='coerce')
data.dropna(inplace=True)
data["min"] = data["min"].astype(int)

# Make 50-50
burned_1 = data[data['min'] == 1]
burned_0 = data[data['min'] == 0]

entries_to_remove = burned_1.shape[0] - burned_0.shape[0]
burned_1_keep = burned_1.sample(burned_0.shape[0])
df = pd.concat([burned_1_keep, burned_0])

X = data.iloc[:, :-1]
Y = data.iloc[:, -1]

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, random_state=0)
classifier = LogisticRegression(solver='lbfgs', random_state=0)
classifier.fit(X_train, Y_train)

predicted_y = classifier.predict(X_test)

print('Accuracy: {:.2f}'.format(classifier.score(X_test, Y_test)))

#ROC Curve
y_pred_proba = classifier.predict_proba(X_test)[::,1]
fpr, tpr, thresholds = roc_curve(Y_test,  y_pred_proba)
# auc = roc_auc_score(Y_test, y_pred_proba)

roc_auc = auc(fpr, tpr)
print("Area under the ROC curve : %f" % roc_auc)

i = np.arange(len(tpr))

roc = pd.DataFrame({'fpr' : pd.Series(fpr, index=i),
                    'tpr' : pd.Series(tpr, index = i),
                    '1-fpr' : pd.Series(1-fpr, index = i),
                    'tf' : pd.Series(tpr - (1-fpr), index = i),
                    'thresholds' : pd.Series(thresholds, index = i),
                    'youden-j': pd.Series(tpr, index = i) - pd.Series(fpr, index=i)})

locroc = roc.iloc[(roc.tf-0).abs().argsort()[:1]]

# plt.plot(fpr,tpr,label="data 1, auc="+str(round(auc,3)))
# plt.legend(loc=4)
# plt.show()

fig, ax = plt.subplots(1, 2, figsize=(15, 5))
ax[0].plot(roc['tpr'])
ax[0].plot(roc['1-fpr'], color = 'red')
ax[0].set_xlabel('1-False Positive Rate')
ax[0].set_ylabel('True Positive Rate')
ax[0].title.set_text('Receiver operating characteristic')
ax[0].set_xticklabels([])


df_test = pd.DataFrame({
    "x": X_test.values.flatten(),
    "y": Y_test.values.flatten(),
    "proba": y_pred_proba
})

final = df_test.loc[df_test['proba']==locroc["thresholds"].values[0]]


df_test.sort_values(by="proba", inplace=True)
df_test.plot(
    x="x", y="proba", ax=ax[1],
    ylabel="Predicted Probabilities", xlabel="X Feature",
    title="Cut-Off", legend=False
)

# sns.regplot(x=X_test, y=y_pred_proba, ax=ax[1],
#             data=pd.DataFrame({'magn': X_test, 'pre_proba': y_pred_proba}),
#             logistic=True,
#             ci=None)
# plt.show()