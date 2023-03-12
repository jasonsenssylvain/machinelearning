import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import statsmodels.api as sm
from statsmodels.graphics.tsaplots import acf,pacf,plot_acf,plot_pacf
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.stattools import adfuller as ADF

plt.ticklabel_format(style='plain')

excel_df = pd.read_excel('./data/GDP.xlsx')
df = pd.DataFrame(excel_df, columns=['年份', 'GDP(亿元)'])
df = df.sort_index(axis = 0, ascending=True)

time_series = pd.Series(df.iloc[:,1].values)
time_series.index = pd.Index(df.iloc[:,0].values)
# time_series.plot(figsize=(12,8))
# plt.show()

time_series = np.log(time_series)
# time_series.plot(figsize=(8,6))
# plt.show()

t = sm.tsa.stattools.adfuller(time_series)
output=pd.DataFrame(index=['Test Statistic Value', "p-value", "Lags Used", "Number of Observations Used","Critical Value(1%)","Critical Value(5%)","Critical Value(10%)"],columns=['value'])
output['value']['Test Statistic Value'] = t[0]
output['value']['p-value'] = t[1]
output['value']['Lags Used'] = t[2]
output['value']['Number of Observations Used'] = t[3]
output['value']['Critical Value(1%)'] = t[4]['1%']
output['value']['Critical Value(5%)'] = t[4]['5%']
output['value']['Critical Value(10%)'] = t[4]['10%']
print(output)

## 明显不平稳
## 继续做差分处理

time_series = time_series.diff(1)
time_series = time_series.dropna(how=any)
# time_series.plot(figsize=(8, 6))
# plt.show()

t = sm.tsa.stattools.adfuller(time_series)
output=pd.DataFrame(index=['Test Statistic Value', "p-value", "Lags Used", "Number of Observations Used","Critical Value(1%)","Critical Value(5%)","Critical Value(10%)"],columns=['value'])
output['value']['Test Statistic Value'] = t[0]
output['value']['p-value'] = t[1]
output['value']['Lags Used'] = t[2]
output['value']['Number of Observations Used'] = t[3]
output['value']['Critical Value(1%)'] = t[4]['1%']
output['value']['Critical Value(5%)'] = t[4]['5%']
output['value']['Critical Value(10%)'] = t[4]['10%']
print(output)

# 已经平稳了

# plot_acf(time_series)
# plot_pacf(time_series)
# plt.show()

# 去 p = 1，去 q = 2
r,rac,Q = sm.tsa.acf(time_series, qstat=True)
prac = pacf(time_series,method='ywmle')
table_data = np.c_[range(1,len(r)), r[1:],rac,prac[1:len(rac)+1],Q]
table = pd.DataFrame(table_data, columns=['lag', "AC","Q", "PAC", "Prob(>Q)"])

print(table)

p,d,q = (1,1,2)
arma_mod = sm.tsa.ARIMA(time_series,order=(p,d,q)).fit()
summary = (arma_mod.summary(alpha=.05))
print(summary)

resid = arma_mod.resid
t = sm.tsa.stattools.adfuller(resid)
output=pd.DataFrame(index=['Test Statistic Value', "p-value", "Lags Used", "Number of Observations Used","Critical Value(1%)","Critical Value(5%)","Critical Value(10%)"],columns=['value'])
output['value']['Test Statistic Value'] = t[0]
output['value']['p-value'] = t[1]
output['value']['Lags Used'] = t[2]
output['value']['Number of Observations Used'] = t[3]
output['value']['Critical Value(1%)'] = t[4]['1%']
output['value']['Critical Value(5%)'] = t[4]['5%']
output['value']['Critical Value(10%)'] = t[4]['10%']
print(output)

arma_model = sm.tsa.ARIMA(resid,order=(0,1,1)).fit()
predict_data = arma_model.predict(start=2015, end=2022, dynamic = False)
print(predict_data)

# 获取拟合时得到的残差序列
fitted_values = arma_model.fittedvalues
print(fitted_values)

# 计算预测时得到的残差序列
forecasted_residuals = predict_data - fitted_values.iloc[-1]
print(forecasted_residuals)

# 计算原始数据的预测值
predicted_values = fitted_values + forecasted_residuals
print(predicted_values.dropna(how=any))