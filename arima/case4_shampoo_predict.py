import pandas as pd
from statsmodels.tsa.stattools import adfuller
import matplotlib.pyplot as plt
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
import statsmodels.api as sm

df = pd.read_csv(r'./data/shampoo_dataset.csv')
# plt.plot(df.Month, df.Sales)
# plt.xticks(rotation=90)
# plt.show()
print(df)

# ARIMA order (p,d,q)
model = sm.tsa.ARIMA(df.Sales, order=(1,1,2))
model_fit = model.fit()
print(model_fit.summary())

predict_seq = model_fit.predict()
# plt.show()
print(predict_seq)

plt.plot(predict_seq, color='red', label='predict_seq')
plt.plot(df.Sales, color='blue', label='real')
plt.show()