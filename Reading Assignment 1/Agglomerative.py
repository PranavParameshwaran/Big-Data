import numpy as np
import matplotlib.pyplot as plt

X = np.array([(2,2),(4,4),(5,5),(6,6),(8,8),(9,9),(10,4),(4,0)])
labels = range(1, 9)
plt.figure(figsize=(10, 7))
plt.subplots_adjust(bottom=0.1)
plt.scatter(X[:,0],X[:,1], label='True Position')

for label, x, y in zip(labels, X[:, 0], X[:, 1]):
	plt.annotate(label,xy=(x, y), xytext=(-3, 3),textcoords='offset points', ha='right', va='bottom')
plt.show()

# Plotting the Dendogram
from scipy.cluster.hierarchy import dendrogram, linkage

linked = linkage(X,'single')
labelList = range(1,9)

plt.figure(figsize=(10,7))
dendrogram(linked,orientation="top",labels=labelList,distance_sort='descending',show_leaf_counts=True)
plt.show()
#########

# Agglomerative Clustering
from sklearn.cluster import AgglomerativeClustering

cluster = AgglomerativeClustering(n_clusters=2,affinity='euclidean',linkage='average')
cluster.fit_predict(X)
print(cluster.labels_)
plt.scatter(X[:,0],X[:,1],c=cluster.labels_,cmap='rainbow')
plt.show()