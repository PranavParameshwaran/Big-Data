import statistics

MidSem = [22,21,28,11,17,19,23,24,23,21,15,14,29,27,29,18,25,26,10,8]
EndSem = [48,45,28,30,46,33,31,29,24,44,47,34,39,37,47,49,43,20,25,28]
Assignment = [16,15,13,17,17,19,13,14,13,11,5,7,9,18,12,8,19,14,10,10]
Total = []

for i in range(20):
	# MidSem.append(input("Enter MidSem(30) "))
	# #print(type(MidSem[i]))
	# EndSem.append(input("Enter EndSem(50) "))
	# Assignment.append(input("Enter Assignment(20) "))
	Total.append(MidSem[i]+EndSem[i]+Assignment[i])

avg = statistics.mean(Total)
GradeCount = {'S':0,'A':0,'B':0,'C':0,'D':0,'E':0,'U':0}
Grade = []
# i = 0
print("Student\tTotal\tGrades",avg)

for i in range(20):
	if Total[i] >= 90:
		Grade.append('S')
		GradeCount['S'] += 1
	elif Total[i] >= 80:
		Grade.append('A')
		GradeCount['A'] += 1
	elif Total[i] >= 70:
		Grade.append('B')
		GradeCount['B'] += 1
	elif Total[i] >= 60:
		Grade.append('C')
		GradeCount['C'] += 1
	elif Total[i] >= 50:
		Grade.append('D')
		GradeCount['D'] += 1
	elif Total[i] >= 0.5*avg:
		Grade.append('E')
		GradeCount['E'] += 1
	else:
		Grade.append('U')
		GradeCount['U'] += 1

		

	print(i+1,'\t',Total[i],'\t',Grade[i])

for i in GradeCount:
	print(i,GradeCount[i])