import statistics

marks = []
for i in range(18):
	if i%2 != 0:
		marks.append(25+((i+7)%10))
	elif i%2 == 0:
		marks.append(25+((i+8)%10))
	print("CSE20D"+str(i+1),marks[i])

mean = statistics.mean(marks)
median = statistics.median(marks)
print("MEAN = ",mean," median = ",median)
