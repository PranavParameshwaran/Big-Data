import matplotlib.pyplot as plt
import numpy as np
# Pie chart, where the slices will be ordered and plotted counter-clockwise:
labels = ('Study', 'Sleep', 'Play', 'Hobby','Family Time')
sizes = [33, 30, 18, 5, 14]
explode = (0, 0, 0, 0, 0)  # only "explode" the 2nd slice (i.e. 'Hogs')

fig1, ax1 = plt.subplots()
ax1.pie(sizes, explode=explode, labels=labels, autopct='%1.1f%%',
        shadow=True, startangle=90)
ax1.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.

plt.show()