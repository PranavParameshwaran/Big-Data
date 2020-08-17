import matplotlib.pyplot as plt
import numpy as np
# Pie chart, where the slices will be ordered and plotted counter-clockwise:
labels = ('S', 'B', 'C', 'D')
sizes = [31, 15, 25, 29]
explode = (0, 0, 0, 0)  # only "explode" the 2nd slice (i.e. 'Hogs')

fig1, ax1 = plt.subplots()
ax1.pie(sizes, explode=explode, labels=labels, autopct='%1.1f%%',
        shadow=True, startangle=90)
ax1.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.

plt.show()

y_pos = np.arange(len(labels))
plt.bar(y_pos, sizes, align='center', alpha=0.5)
plt.xticks(y_pos, labels)
plt.ylabel('Number of Students')
plt.title('Grades')

plt.show()