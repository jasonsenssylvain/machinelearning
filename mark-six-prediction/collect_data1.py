import os
import requests
import json

path = './ssq1.txt'

if os.path.exists(path):
    os.remove(path)
else:
    print("not file！")

def save_to_file(content):
    with open(path, 'a', encoding='utf-8') as f:
        f.write(content + '\n')

for year in range(2022, 1995, -1):
	print("get data for year: " + str(year))
	url = 'https://1680660.com/smallSix/findSmallSixHistory.do?year=' + str(year) + '&type=1'
	r = requests.post(url)
	result = json.loads(r.text)
	# print(result)
	for data in result['result']['data']['bodyList']:
		dateInfo = data['preDrawDate']
		issueInfo = data['issue']
		numInfo = data['preDrawCode']

		data = dateInfo + ", " + str(year) + str(issueInfo) + ", " + numInfo
		save_to_file(data)




