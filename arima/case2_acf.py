import matplotlib.pyplot as plt
import pandas as pd
import statsmodels.api as sm
from statsmodels.tsa.arima_model import ARIMA
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.stattools import adfuller as ADF

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