import matplotlib.pyplot as plt
import pandas as pd
from statsmodels.tsa.seasonal import seasonal_decompose


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
