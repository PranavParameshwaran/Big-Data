from sklearn.datasets.samples_generator import make_blobs
from sklearn.cluster import Birch

from matplotlib import pyplot as plt

X, clusters = make_blobs(n_samples=450, centers=6, cluster_std=0.7, random_state=0)
plt.scatter(X[:,0], X[:,1], alpha=0.7, edgecolors='b')
plt.show()
brc = Birch(branching_factor=50, n_clusters=None, threshold=1.5)
brc.fit(X)

labels = brc.predict(X)
plt.scatter(X[:,0], X[:,1], c=labels, cmap='seismic', alpha=0.7, edgecolors='b')
plt.show()