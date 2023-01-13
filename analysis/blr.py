"""
Binomial logistic Regression
"""
import pandas as pd
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression

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
# count_no_sub = len(data[data['min'] == 0])
# count_sub = len(data[data['min'] == 1])
# pct_of_no_sub = count_no_sub / (count_no_sub + count_sub)
# print("percentage of no subscription is", pct_of_no_sub * 100)
# pct_of_sub = count_sub / (count_no_sub + count_sub)
# print("percentage of subscription", pct_of_sub * 100)

# data.zone.hist()
# plt.title('RBR - Histogram of Magnitude')
# plt.xlabel('Magnitude')
# plt.ylabel('Frequency')
# plt.savefig('hist_magn')

Zones_X = data.iloc[:, :-1].values
Burn_Y = data.iloc[:, -1].values

Zones_train, Zones_test, Burn_train, Burn_test = \
    train_test_split(Zones_X, Burn_Y, test_size=0.05, random_state=1, stratify=Burn_Y)

log_reg = LogisticRegression(solver='sag', penalty='none', max_iter=200000)
log_reg.fit(Zones_train, Burn_train)

# make predictions on the test data
y_pred = log_reg.predict(Zones_test)

# calculate the accuracy of the model
accuracy = log_reg.score(Zones_test, Burn_test)

print(f'Accuracy: {accuracy:.2f}')

# get the coefficient of the magnitude feature
magnitude_coef = log_reg.coef_[0][0]

# calculate the threshold
threshold = -log_reg.intercept_[0] / magnitude_coef

print(f'Optimal magnitude threshold: {threshold}')
