import pandas as pd
import numpy as np

from sklearn.preprocessing import StandardScaler

from keras.models import Sequential
from keras.layers import LSTM, Dense, Bidirectional, Dropout

from tensorflow import keras

model = keras.models.load_model('mymodel_txt')

PATH = r"ssq1.txt"
ssq = open(PATH)
lines = ssq.readlines()
# print(lines)

history = []
for line in lines:
	data = line.split(',')
	dataArray = []

	if data[0].startswith('2022') or data[0].startswith('2023'):
		dataArray.append(data[0])
		dataArray.append(data[1])
		for i in range(2, len(data)):
			num = data[i].strip()
			num = num.replace("\n", "")
			num = int(num)
			dataArray.append(num)
		history.append(dataArray)

# print(history)
history.reverse()
new_df = pd.DataFrame(np.array(history),
	columns=list('abABCDEFG'))

new_df[['A', 'B', 'C', 'D', 'E', 'F', 'G']] = new_df[['A', 'B', 'C', 'D', 'E', 'F', 'G']].apply(pd.to_numeric)
print(new_df)

new_df['生肖1'] = new_df['A'].mod(12)
new_df['生肖2'] = new_df['B'].mod(12)
new_df['生肖3'] = new_df['C'].mod(12)
new_df['生肖4'] = new_df['D'].mod(12)
new_df['生肖5'] = new_df['E'].mod(12)
new_df['生肖6'] = new_df['F'].mod(12)
new_df['生肖7'] = new_df['G'].mod(12)
new_df['生肖个数'] = 0

new_df['预测1'] = 0
new_df['预测2'] = 0
new_df['预测3'] = 0
new_df['预测4'] = 0
new_df['预测5'] = 0
new_df['预测6'] = 0
new_df['预测7'] = 0
new_df['预测生肖数量'] = 0
new_df['预测命中数量'] = 0

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


## 开始预测
for i in range(0, number_of_rows - window_length):
	curr_df = new_df.iloc[i : i + window_length]
	new_curr_data_df = pd.DataFrame(curr_df, columns=['生肖1', '生肖2', '生肖3', '生肖4', '生肖5', '生肖6', '生肖7'])
	scaled_to_predict = scaler.transform(new_curr_data_df.values)
	scaled_output = model.predict(np.array([scaled_to_predict]))
	result_predict = scaler.inverse_transform(scaled_output).astype(int)[0]

	# 统计生肖个数
	data = {}
	data[new_df.loc[i + window_length, '生肖1']] = new_df.loc[i + window_length, '生肖1']
	data[new_df.loc[i + window_length, '生肖2']] = new_df.loc[i + window_length, '生肖2']
	data[new_df.loc[i + window_length, '生肖3']] = new_df.loc[i + window_length, '生肖3']
	data[new_df.loc[i + window_length, '生肖4']] = new_df.loc[i + window_length, '生肖4']
	data[new_df.loc[i + window_length, '生肖5']] = new_df.loc[i + window_length, '生肖5']
	data[new_df.loc[i + window_length, '生肖6']] = new_df.loc[i + window_length, '生肖6']
	data[new_df.loc[i + window_length, '生肖7']] = new_df.loc[i + window_length, '生肖7']

	new_df.loc[i + window_length, '生肖个数'] = len(data)

	new_df.loc[i + window_length, '预测1'] = result_predict[0]
	new_df.loc[i + window_length, '预测2'] = result_predict[1]
	new_df.loc[i + window_length, '预测3'] = result_predict[2]
	new_df.loc[i + window_length, '预测4'] = result_predict[3]
	new_df.loc[i + window_length, '预测5'] = result_predict[4]
	new_df.loc[i + window_length, '预测6'] = result_predict[5]
	new_df.loc[i + window_length, '预测7'] = result_predict[6]


	predict = {}
	predict[new_df.loc[i + window_length, '预测1']] = result_predict[0]
	predict[new_df.loc[i + window_length, '预测2']] = result_predict[1]
	predict[new_df.loc[i + window_length, '预测3']] = result_predict[2]
	predict[new_df.loc[i + window_length, '预测4']] = result_predict[3]
	predict[new_df.loc[i + window_length, '预测5']] = result_predict[4]
	predict[new_df.loc[i + window_length, '预测6']] = result_predict[5]
	predict[new_df.loc[i + window_length, '预测7']] = result_predict[6]

	new_df.loc[i + window_length, '预测生肖数量'] = len(predict)


	predictMatch = 0
	for predictValue in predict.values():
		if predictValue in data:
			predictMatch = predictMatch + 1

	new_df.loc[i + window_length, '预测命中数量'] = predictMatch / len(predict)







reverse_curr_df = new_df.iloc[::-1]
reverse_curr_df.to_excel(r"data/result_txt.xlsx")

# reverse_curr_df = new_df.iloc[::-1]
# result = "日期 \t 期号 \t 号码1 \t 号码2 \t 号码3 \t 号码4 \t"
# for i in range(0, reverse_curr_df.values.shape[0]):
# 	result = result + reverse_curr_df.loc[i, 'a'] + ", "
# 	result = result + reverse_curr_df.loc[i, 'b'] + ", "
#
# 	result = result + str(reverse_curr_df.loc[i, 'A']) + ", "
# 	result = result + str(reverse_curr_df.loc[i, 'B']) + ", "
# 	result = result + str(reverse_curr_df.loc[i, 'C']) + ", "
# 	result = result + str(reverse_curr_df.loc[i, 'D']) + ", "
# 	result = result + str(reverse_curr_df.loc[i, 'E']) + ", "
# 	result = result + str(reverse_curr_df.loc[i, 'F']) + ", "
# 	result = result + str(reverse_curr_df.loc[i, 'G']) + ", "
#
# 	result = result + str(reverse_curr_df.loc[i, '生肖1']) + ", "
# 	result = result + str(reverse_curr_df.loc[i, '生肖2']) + ", "
# 	result = result + str(reverse_curr_df.loc[i, '生肖3']) + ", "
# 	result = result + str(reverse_curr_df.loc[i, '生肖4']) + ", "
# 	result = result + str(reverse_curr_df.loc[i, '生肖5']) + ", "
# 	result = result + str(reverse_curr_df.loc[i, '生肖6']) + ", "
# 	result = result + str(reverse_curr_df.loc[i, '生肖7']) + ", "
#
# 	result = result + str(reverse_curr_df.loc[i, '预测1']) + ", "
# 	result = result + str(reverse_curr_df.loc[i, '预测2']) + ", "
# 	result = result + str(reverse_curr_df.loc[i, '预测3']) + ", "
# 	result = result + str(reverse_curr_df.loc[i, '预测4']) + ", "
# 	result = result + str(reverse_curr_df.loc[i, '预测5']) + ", "
# 	result = result + str(reverse_curr_df.loc[i, '预测6']) + ", "
# 	result = result + str(reverse_curr_df.loc[i, '预测7']) + ", "
#
# 	result = result + "\n"
#
# path = './result1.txt'
# with open(path, 'a', encoding='utf-8') as f:
# 	f.write(result + '\n')