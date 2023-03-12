import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.stattools import adfuller

df = pd.read_csv(r'./data/website_data.csv')
print(df.info())
# df.plot()
# plt.show()

df = np.log(df)
# df.plot()
# plt.show()

split_count = 30
msk = df.index < len(df) - split_count
print(msk)
df_train = df[msk].copy()
df_tset = df[~msk].copy()

# fig, axes = plt.subplots(1, 2, sharex=False)
# acf_original = plot_acf(df_train)
# pacf_original = plot_pacf(df_train)
# plt.show()

# ADF test
result = adfuller(df_train)
output = pd.DataFrame(index=['Test Statistic Value', "p-value", "Lags Used", "Number of Observations Used","Critical Value(1%)","Critical Value(5%)","Critical Value(10%)"], columns=['value'])
output['value']['Test Statistic Value'] = result[0]
output['value']['p-value'] = result[1]
output['value']['Lags Used'] = result[2]
output['value']['Number of Observations Used'] = result[3]
output['value']['Critical Value(1%)'] = result[4]['1%']
output['value']['Critical Value(5%)'] = result[4]['5%']
output['value']['Critical Value(10%)'] = result[4]['10%']
print(output)
# 结果不行

df_train_diff = df_train.diff().dropna()
# df_train_diff.plot()
# plt.show()
# acf_original = plot_acf(df_train_diff)
# plt.show()
# pacf_original = plot_pacf(df_train_diff)
# plt.show()

result = adfuller(df_train_diff)
output = pd.DataFrame(index=['Test Statistic Value', "p-value", "Lags Used", "Number of Observations Used","Critical Value(1%)","Critical Value(5%)","Critical Value(10%)"], columns=['value'])
output['value']['Test Statistic Value'] = result[0]
output['value']['p-value'] = result[1]
output['value']['Lags Used'] = result[2]
output['value']['Number of Observations Used'] = result[3]
output['value']['Critical Value(1%)'] = result[4]['1%']
output['value']['Critical Value(5%)'] = result[4]['5%']
output['value']['Critical Value(10%)'] = result[4]['10%']
print(output)

