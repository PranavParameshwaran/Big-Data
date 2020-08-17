import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

age = [7, 9, 27, 28, 55, 45, 34, 65, 54, 67, 34, 23, 24, 66, 53, 45, 44, 88, 22, 33, 55, 35, 33, 37, 47, 41,31, 30,
29, 12]

data = np.array(age)
data = pd.DataFrame(data = data)
sns.distplot(data,kde = True, bins = 8, rug = True, color = 'red')

plt.show()

# Ans = Single Modal 
