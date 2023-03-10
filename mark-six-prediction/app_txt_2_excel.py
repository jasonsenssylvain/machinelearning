import pandas as pd
import numpy as np

PATH = r"ssq1.txt"
ssq = open(PATH)
lines = ssq.readlines()

history = []
for line in lines:
	data = line.split(',')
	dataArray = []

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

new_df.to_excel(r"ssq1.xlsx")