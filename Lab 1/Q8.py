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

# Total = [20,30,40,78,70,65,65,88,87,78,76,68,45,69,93,23,45,78,97,78]
Mean = statistics.mean(Total)
PassingMin = Mean*0.5
Sum = 0
Count = 0
for i in range(20):
	if Total[i] >= PassingMin:
		Sum += Total[i]
		Count += 1
PassingStudMean = Sum/Count
X = PassingStudMean - PassingMin
MaxMark = max(Total)
Scutoff = MaxMark -0.1*(MaxMark - PassingStudMean)
Y = Scutoff - PassingStudMean
Acutoff = (5*Y)/8+PassingStudMean
Bcutoff = (2*Y)/8+PassingStudMean
Ccutoff = PassingStudMean - ((2*X)/8)
Dcutoff = PassingStudMean - ((5*X)/8)
Ecutoff = PassingMin

GradeCount = {'S':0,'A':0,'B':0,'C':0,'D':0,'E':0,'U':0}
Grade = []
# i = 0
print("Student\tTotal\tGrades")

for i in range(20):
	if Total[i] >= Scutoff:
		Grade.append('S')
		GradeCount['S'] += 1
	elif Total[i] >= Acutoff:
		Grade.append('A')
		GradeCount['A'] += 1
	elif Total[i] >= Bcutoff:
		Grade.append('B')
		GradeCount['B'] += 1
	elif Total[i] >= Ccutoff:
		Grade.append('C')
		GradeCount['C'] += 1
	elif Total[i] >= Dcutoff:
		Grade.append('D')
		GradeCount['D'] += 1
	elif Total[i] >= Ecutoff:
		Grade.append('E')
		GradeCount['E'] += 1
	else:
		Grade.append('U')
		GradeCount['U'] += 1

		

	print(i+1,'\t',Total[i],'\t',Grade[i])

for i in GradeCount:
	print(i,GradeCount[i])

# print("\n\n\n")
# print("PassingMin",PassingMin,"PassingStudMean",PassingStudMean,"X",X,"Bcutoff",Bcutoff)