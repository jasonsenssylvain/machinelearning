import pandas as pd
import numpy as np

from sklearn.preprocessing import StandardScaler

from keras.models import Sequential
from keras.layers import LSTM, Dense, Bidirectional, Dropout

# import collect_data1

PATH = r"data/Mark-Six-Data-small.xlsx"
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

new_data_df2 = new_data_df

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

train = np.empty([number_of_rows - window_length, window_length, number_of_features], dtype=float)
label = np.empty([number_of_rows - window_length, number_of_features], dtype=float)

for i in range(0, number_of_rows - window_length):
	train[i]=transformed_df.iloc[i:i+window_length, 0: number_of_features]
	label[i]=transformed_df.iloc[i+window_length: i + window_length + 1, 0: number_of_features]

print("train: ", train.shape)
print("label: ", label.shape)

batch_size = 100

model = Sequential()
model.add(Bidirectional(LSTM(240,
                             input_shape=(window_length, number_of_features),
                             return_sequences=True)))
model.add(Dropout(0.2))
model.add(Bidirectional(LSTM(240,
                             input_shape=(window_length, number_of_features),
                             return_sequences=True)))
model.add(Dropout(0.2))
model.add(Bidirectional(LSTM(240,
                             input_shape=(window_length, number_of_features),
                             return_sequences=True)))
model.add(Bidirectional(LSTM(240,
                             input_shape=(window_length, number_of_features),
                             return_sequences=False)))
model.add(Dense(59))
model.add(Dense(number_of_features))
model.compile(loss='mse', optimizer='rmsprop', metrics=['accuracy'])

model.fit(train, label, batch_size=batch_size, epochs=100)

model.save('mymodel')

## 开始预测

# row_diff_count = total_number_of_rows - number_of_rows
# print("row_diff_count: ", row_diff_count)
#
# for i in range(0, row_diff_count):
# 	curr_df = excel_data_df.iloc[(row_diff_count - i) :(row_diff_count - i) + window_length, 0: number_of_features]
# 	reverse_curr_df = curr_df.iloc[::-1]
# 	print(reverse_curr_df)