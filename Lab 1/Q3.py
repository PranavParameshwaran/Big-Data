import random
import statistics
y = []
for i in range(20):
	x = random.randint(6,100)
	y.append((2*x)+3) 
	print(x,y[i])

stddev = statistics.stdev(y)
print("Standard Deviation",stddev)