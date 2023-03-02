import pandas as pd
import numpy as np

from sklearn.preprocessing import StandardScaler

from keras.models import Sequential
from keras.layers import LSTM, Dense, Bidirectional, Dropout

# import collect_data1

PATH = r"ssq1-small.txt"
ssq = open(PATH)
lines = ssq.readlines()
# print(lines)

history = []
for line in lines:
	data = line.split(',')
	dataArray = []
	del(data[0])
	del(data[0])
	for num in data:
		num = num.strip()
		num = num.replace("\n", "")
		num = int(num)
		dataArray.append(num)
	history.append(dataArray)

# print(history)
history.reverse()
new_df = pd.DataFrame(np.array(history),
	columns=list('ABCDEFG'))

new_df['生肖1'] = new_df['A'].mod(12)
new_df['生肖2'] = new_df['B'].mod(12)
new_df['生肖3'] = new_df['C'].mod(12)
new_df['生肖4'] = new_df['D'].mod(12)
new_df['生肖5'] = new_df['E'].mod(12)
new_df['生肖6'] = new_df['F'].mod(12)
new_df['生肖7'] = new_df['G'].mod(12)

new_df['预测1'] = 0
new_df['预测2'] = 0
new_df['预测3'] = 0
new_df['预测4'] = 0
new_df['预测5'] = 0
new_df['预测6'] = 0
new_df['预测7'] = 0

print(new_df.head())

df = pd.DataFrame(new_df, columns=['生肖1', '生肖2', '生肖3', '生肖4', '生肖5', '生肖6', '生肖7'])

scaler = StandardScaler().fit(df.values)
transformed_dataset = scaler.transform(df.values)
transformed_df = pd.DataFrame(data=transformed_dataset, index=df.index)
print(transformed_df)

number_of_rows = df.values.shape[0]
print("number_of_rows: ", number_of_rows)

window_length = 5  # amount of past games we need to take in consideration for prediction
number_of_features = df.values.shape[1]
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

currData = history[-window_length:]
print(currData)

## start predict
model.save('mymodel_txt')
