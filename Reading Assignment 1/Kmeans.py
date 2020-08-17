import numpy as np
# import pandas as pd
from matplotlib import pyplot as plt
from sklearn.datasets.samples_generator import make_blobs
from sklearn.cluster import KMeans

# Initialization of Dataset
# X = np.array([(2,2),(4,4),(5,5),(6,6),(8,8),(9,9),(10,4),(4,0)])
X = np.array([(1,1),(1,2),(2,3),(2,1),(-1,1),(3,-1),(2,-1),(-2,0)])
plt.scatter(X[:,0], X[:,1])
plt.show()


def distance(x1,x2,y1,y2):
	d = ((x1 - x2)**2) + ((y1 - y2)**2)
	return d  


W1 = []
W2 = []
for x,y in X:
	if distance(x,3.75,y,2.75) < distance(x,8.25,y,6.75):
		W1.append((x,y))
	else:
		W2.append((x,y))
print(W1)
print(W2)

# ELBOW METHOD

#

kmeans = KMeans(n_clusters=2, init='k-means++', max_iter=300, n_init=8, random_state=0)
pred_y = kmeans.fit_predict(X)
plt.scatter(X[:,0], X[:,1])
plt.scatter(kmeans.cluster_centers_[:,0],kmeans.cluster_centers_[:,1],cmap='rainbow')
plt.show()