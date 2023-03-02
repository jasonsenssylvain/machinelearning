import pandas as pd
import numpy as np

from sklearn.preprocessing import StandardScaler

from keras.models import Sequential
from keras.layers import LSTM, Dense, Bidirectional, Dropout

from tensorflow import keras

model = keras.models.load_model('mymodel')


PATH = r"data/Mark-Six-Data.xlsx"
excel_data_df = pd.read_excel(PATH)
print(excel_data_df)

excel_data_df['new1'] = excel_data_df['码 1'].mod(12)
excel_data_df['new2'] = excel_data_df['码 2'].mod(12)
excel_data_df['new3'] = excel_data_df['码 3'].mod(12)
excel_data_df['new4'] = excel_data_df['码 4'].mod(12)
excel_data_df['new5'] = excel_data_df['码 5'].mod(12)
excel_data_df['new6'] = excel_data_df['码 6'].mod(12)
excel_data_df['new7'] = excel_data_df['码 7'].mod(12)

new_data_df = pd.DataFrame(excel_data_df, columns=['new1', 'new2', 'new3', 'new4', 'new5', 'new6', 'new7'])
# print(new_data_df)
total_number_of_rows = new_data_df.values.shape[0]
print("total number_of_rows: ", total_number_of_rows)

new_data_df2 = new_data_df.head(2070)

reverse_data_df = new_data_df2.iloc[::-1]
print(reverse_data_df)

scaler = StandardScaler().fit(reverse_data_df.values)
transformed_dataset = scaler.transform(reverse_data_df.values)
transformed_df = pd.DataFrame(data=transformed_dataset, index=reverse_data_df.index)
print(transformed_df)


number_of_rows = reverse_data_df.values.shape[0]
print("number_of_rows: ", number_of_rows)

window_length = 7  # amount of past games we need to take in consideration for prediction
number_of_features = reverse_data_df.values.shape[1]
print("number_of_features: ", number_of_features)

## 开始预测

row_diff_count = total_number_of_rows - number_of_rows
print("row_diff_count: ", row_diff_count)

excel_data_df['p1'] = 0
excel_data_df['p2'] = 0
excel_data_df['p3'] = 0
excel_data_df['p4'] = 0
excel_data_df['p5'] = 0
excel_data_df['p6'] = 0
excel_data_df['p7'] = 0

for i in range(0, row_diff_count):
	curr_df = excel_data_df.iloc[(row_diff_count - i) :(row_diff_count - i) + window_length]
	print(curr_df)
	new_curr_data_df = pd.DataFrame(curr_df, columns=['new1', 'new2', 'new3', 'new4', 'new5', 'new6', 'new7'])
	reverse_curr_df = new_curr_data_df.iloc[::-1]
	scaled_to_predict = scaler.transform(reverse_curr_df.values)
	scaled_output = model.predict(np.array([scaled_to_predict]))
	result_predict = scaler.inverse_transform(scaled_output).astype(int)[0]
	print(result_predict)
	excel_data_df.loc[row_diff_count - i - 1, 'p1'] = result_predict[0]
	excel_data_df.loc[row_diff_count - i - 1, 'p2'] = result_predict[1]
	excel_data_df.loc[row_diff_count - i - 1, 'p3'] = result_predict[2]
	excel_data_df.loc[row_diff_count - i - 1, 'p4'] = result_predict[3]
	excel_data_df.loc[row_diff_count - i - 1, 'p5'] = result_predict[4]
	excel_data_df.loc[row_diff_count - i - 1, 'p6'] = result_predict[5]
	excel_data_df.loc[row_diff_count - i - 1, 'p7'] = result_predict[6]
	print(excel_data_df.loc[row_diff_count - i - 1])

excel_data_df.to_excel(r"data/result.xlsx")


