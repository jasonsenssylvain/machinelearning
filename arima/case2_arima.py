import matplotlib.pyplot as plt
import pandas as pd
import statsmodels.api as sm
from statsmodels.tsa.arima_model import ARIMA
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.stattools import adfuller as ADF


def diff(timeseries):
	timeseries_diff1 = timeseries.diff(1)
	timeseries_diff2 = timeseries_diff1.diff(1)

	timeseries_diff1 = timeseries_diff1.fillna(0)
	timeseries_diff2 = timeseries_diff2.fillna(0)

	timeseries_adf = ADF(timeseries['value'].tolist())
	timeseries_diff1_adf = ADF(timeseries_diff1['value'].tolist())
	timeseries_diff2_adf = ADF(timeseries_diff2['value'].tolist())

	print(timeseries)
	print('timeseries_adf : ', timeseries_adf)
	print('timeseries_diff1_adf : ', timeseries_diff1_adf)
	print('timeseries_diff2_adf : ', timeseries_diff2_adf)

	plt.figure(figsize=(12, 8))
	plt.plot(timeseries, label='Original', color='blue')
	plt.plot(timeseries_diff1, label='Diff1', color='red')
	plt.plot(timeseries_diff2, label='Diff2', color='purple')
	plt.legend(loc='best')
	plt.show()

dateparse = lambda dates: pd.datetime.strptime(dates, '%Y-%m-%d')
purchase_seq_train = pd.read_csv('./data/purchase_seq_train.csv', parse_dates=['report_date'],
                                 index_col='report_date', date_parser=dateparse)
print(type(purchase_seq_train))
diff(purchase_seq_train)


def autocorrelation(timeseries, lags):
  fig = plt.figure(figsize=(12, 8))
  ax1 = fig.add_subplot(211)
  sm.graphics.tsa.plot_acf(timeseries, lags=lags, ax=ax1)
  ax2 = fig.add_subplot(212)
  sm.graphics.tsa.plot_pacf(timeseries, lags=lags, ax=ax2)
  plt.show()

dateparse = lambda dates: pd.datetime.strptime(dates, '%Y-%m-%d')
purchase_seq_train = pd.read_csv('./data/purchase_seq_train.csv', parse_dates=['report_date'],
                                 index_col='report_date', date_parser=dateparse)

purchase_seq_train_diff = purchase_seq_train.diff(1)
purchase_seq_train_diff = purchase_seq_train_diff.fillna(0)

autocorrelation(purchase_seq_train_diff, 20)



def decomposing(timeseries):
  decomposition = seasonal_decompose(timeseries)
  trend = decomposition.trend
  seasonal = decomposition.seasonal
  residual = decomposition.resid

  plt.figure(figsize=(16, 12))
  plt.subplot(411)
  plt.plot(timeseries, label='Original')
  plt.legend(loc='best')
  plt.subplot(412)
  plt.plot(trend, label='Trend')
  plt.legend(loc='best')
  plt.subplot(413)
  plt.plot(seasonal, label='Seasonarity')
  plt.legend(loc='best')
  plt.subplot(414)
  plt.plot(residual, label='Residual')
  plt.legend(loc='best')
  plt.show()


dateparse = lambda dates: pd.datetime.strptime(dates, '%Y-%m-%d')
purchase_seq_train = pd.read_csv('./data/purchase_seq_train.csv', parse_dates=['report_date'],
                                 index_col='report_date', date_parser=dateparse)

decomposing(purchase_seq_train)


purchase_seq_train = pd.read_csv('./data/purchase_seq_train.csv', parse_dates=['report_date'],
                                 index_col='report_date', date_parser=dateparse)
print(purchase_seq_train)
decomposition = seasonal_decompose(purchase_seq_train)
trend = decomposition.trend
seasonal = decomposition.seasonal
residual = decomposition.resid

trend = trend.fillna(0)
seasonal = seasonal.fillna(0)
residual = residual.fillna(0)

trend_df = trend.to_frame(name='value')
residual_df = residual.to_frame(name='value')

diff(trend_df)
diff(residual_df)

autocorrelation(trend_df, 20)
autocorrelation(residual_df, 20)