import pandas as pd
from statsmodels.tsa.stattools import adfuller
import matplotlib.pyplot as plt

df = pd.read_csv(r'./data/shampoo_dataset.csv')
# plt.plot(df.Month, df.Sales)
# plt.xticks(rotation=90)
# plt.show()
print(df)

#