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

diff(purchase_seq_train)