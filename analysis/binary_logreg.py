import pandas as pd
import numpy as np
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_curve, auc, RocCurveDisplay

import matplotlib.pyplot as plt

import csv
import warnings

warnings.simplefilter(action='ignore', category=FutureWarning)

plt.rc("font", size=14)

sns.set(style="white")
sns.set(style="whitegrid", color_codes=True)

index = input('Choose [nbr, dnbr, rbr]')
index_dict = {'nbr': 'NBR',
              'dnbr': 'dNBR',
              'rbr': 'RBR'}

data = pd.read_csv(
    f"E:/Publications/BFAST_Monitor/results/blr/zonal_statistics/{index}/all_{index}_raslayer_zonal_statistics.csv",
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

classifier = LogisticRegression(C=1,
                                solver='saga',
                                penalty='l1',
                                max_iter=10000, random_state=42, multi_class='ovr')

classifier.fit(X_train, Y_train)

predicted_y = classifier.predict(X_test)

print('Accuracy: {:.2f}'.format(classifier.score(X_test, Y_test)))

# ROC Curve
y_pred_proba = classifier.predict_proba(X_test)[::, 1]
fpr, tpr, thresholds = roc_curve(Y_test, y_pred_proba)

roc_auc = auc(fpr, tpr)
print("Area under the ROC curve : %f" % roc_auc)

i = np.arange(len(tpr))

roc = pd.DataFrame({'fpr': pd.Series(fpr, index=i),
                    'tpr': pd.Series(tpr, index=i),
                    '1-fpr': pd.Series(1 - fpr, index=i),
                    'tf': pd.Series(tpr - (1 - fpr), index=i),
                    'thresholds': pd.Series(thresholds, index=i),
                    'youden-j': pd.Series(tpr, index=i) - pd.Series(fpr, index=i)})

locroc = roc.iloc[(roc.tf - 0).abs().argsort()[:1]]

df_test = pd.DataFrame({
    "x": X_test.values.flatten(),
    "y": Y_test.values.flatten(),
    "proba": y_pred_proba
})

final = df_test.loc[df_test['proba'] == locroc["thresholds"].values[0]]
# final50 = df_test.loc[round(df_test['proba'], 6) == 0.5]
final50 = df_test.loc[(round(df_test['proba'], 5) == 0.5) & df_test['y'] == 1]

pd.concat([final, final50]).to_csv(f'E:/Publications/BFAST_Monitor/results/blr/zonal_statistics/blr_{index}.csv',
                                   sep=',')

df_test.sort_values(by="proba", inplace=True)
index_str = index_dict[index]

with plt.style.context("bmh"):
    fig, ax = plt.subplots(1, 2, figsize=(15, 5))

    ax[0].axline(xy1=(0, 0), slope=1, color="k", ls=":")
    reversed_fpr = pd.Series(data=roc['1-fpr'].values, index=(roc['1-fpr'].index / 114758))
    ax[0].plot(reversed_fpr, color='red')
    RocCurveDisplay(
        fpr=roc.fpr, tpr=roc.tpr,
        roc_auc=roc_auc).plot(ax=ax[0])
    ax[0].set_title(f'{index_str} - ROC curve')

    df_test.plot(
        x="x", y="proba", ax=ax[1],
        ylabel="Predicted Probabilities", xlabel="Magnitude",
        title=f"Cut-Off ({index_str})", legend=False
    )
    ax[1].axvline(final.iloc[0].x, color="r", ls="-.", lw=1)
    ax[1].axhline(final.iloc[0].proba, color="r", ls="-.", lw=1)
    ax[1].axvline(final50.iloc[0].x, color="k", ls=":", lw=1)
    ax[1].axhline(0.5, color="k", ls=":", lw=1)
    for tick in ax[1].get_xticklabels():
        tick.set_visible(True)
    plt.tight_layout()
    plt.show()
    plt.savefig(f'E:/Publications/BFAST_Monitor/results/blr/zonal_statistics/blr_{index}.png',
                dpi=300)

with open(f'E:/Publications/BFAST_Monitor/results/blr/zonal_statistics/blr_{index}.csv', 'a', newline='\n') as res_file:
    writer = csv.writer(res_file)
    writer.writerow(['Accuracy', classifier.score(X_test, Y_test)])
    writer.writerow(['AUC ROC', roc_auc])
