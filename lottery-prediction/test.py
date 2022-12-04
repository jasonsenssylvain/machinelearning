import random




i = 0
text = ""
while(i < 1200):

	if i % 6 == 0:
		text = text + "["
	data = random.randint(1, 61)
	text = text + str(data)

	if i % 6 != 5:
		text = text + ","
	else:
		text = text + "],"
	i = i + 1

print(text)
