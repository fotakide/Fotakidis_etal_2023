"""
Binomial logistic Regression
"""
import pandas as pd
import numpy as np
import seaborn as sns

import statsmodels.api as sm
from statsmodels.formula.api import logit

from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix
from sklearn.metrics import roc_curve, roc_auc_score, auc, accuracy_score

from sklearn.metrics import classification_report, accuracy_score

import matplotlib.pyplot as plt

plt.rc("font", size=14)

sns.set(style="white")
sns.set(style="whitegrid", color_codes=True)

data = pd.read_csv(
    "E:/Publications/BFAST_Monitor/results/blr/zonal_statistics/nbr/all_nbr_raslayer_zonal_statistics.csv",
    usecols=['zone', 'min'], low_memory=False)
data["zone"] = pd.to_numeric(data["zone"], errors='coerce')
data["min"] = pd.to_numeric(data["min"], errors='coerce')
data.dropna(inplace=True)
data["min"] = data["min"].astype(int)
#
# sns.countplot(x='min', data=data, palette='hls')
# plt.show()
# # plt.savefig('count_plot')
#
# count_no_burn = len(data[data['min'] == 0])
# count_burned = len(data[data['min'] == 1])
# pct_of_no_burn = count_no_burn / (count_no_burn + count_burned)
# print("percentage of no burn is", pct_of_no_burn * 100)
# pct_of_sub = count_burned / (count_no_burn + count_burned)
# print("percentage of burned", pct_of_sub * 100)
#
# data.zone.hist()
# plt.title('Histogram of Magnitude')
# plt.xlabel('Magnitude')
# plt.ylabel('Frequency')
#
# data[data['min'] == 1].zone.hist()
# data[data['min'] == 0].zone.hist()
# plt.title('Histogram of Magnitude')
# plt.xlabel('Magnitude')
# plt.ylabel('Frequency')

# plt.savefig('hist_magn')

#Make 50-50
burned_1 = data[data['min'] == 1]
burned_0 = data[data['min'] == 0]

entries_to_remove = burned_1.shape[0] - burned_0.shape[0]
burned_1_keep = burned_1.sample(burned_0.shape[0])
df = pd.concat([burned_1_keep, burned_0])

# sns.countplot(x='min', data=df, palette='hls')
# plt.show()
# plt.savefig('count_plot')


count_no_burn = len(df[df['min'] == 0])
count_burned = len(df[df['min'] == 1])
pct_of_no_burn = count_no_burn / (count_no_burn + count_burned)
print("percentage of no burn is", pct_of_no_burn * 100)
pct_of_sub = count_burned / (count_no_burn + count_burned)
print("percentage of burned", pct_of_sub * 100)

# df.zone.hist()
# plt.title('Histogram of Magnitude')
# plt.xlabel('Magnitude')
# plt.ylabel('Frequency')
#
# df[df['min'] == 1].zone.hist()
# df[df['min'] == 0].zone.hist()
# plt.title('Histogram of Magnitude')
# plt.xlabel('Magnitude')
# plt.ylabel('Frequency')



#

Zones_X = data.iloc[:, :-1].values
Burn_Y = data.iloc[:, -1].values

train_data, test_data = train_test_split(df, test_size=0.25, random_state=0)
# random_state is the size for each split and gives consistent results and reproducibility


formula = ('min~zone')
model = logit(formula=formula,data=train_data).fit()
print(np.exp(model.params))

AME = model.get_margeff(at='overall', method='dydx')
print(AME.summary())

prediction = model.predict(exog=test_data)
cutoff = 0.5
y_prediction = np.where(prediction > cutoff, 1, 0)

y_actual = test_data["min"]

conf_matrix = pd.crosstab(y_actual, y_prediction,
                          rownames=["Actual"],
                          colnames=["predicted"],
                          margins=True)
print(conf_matrix)

accuracy = accuracy_score(y_actual, y_prediction)
print('Accuracy: %.2f' %accuracy)

print(classification_report(y_actual, y_prediction))


#BLR plot
sns.regplot(x=df['zone'], y=df['min'],
            y_jitter=None,
            data=df,
            logistic=True,
            ci=None)
plt.show()