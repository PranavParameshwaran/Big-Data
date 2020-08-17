import numpy as np

def init_medoids(X,k=2):
	from numpy.random import choice
	from numpy.random import seed
	seed(1)
	samples = choice(len(X),size=k,replace=False)
	return X[samples,:]

def compute_d_p(X,medoids,p):
	m = len(X)
	medoids_shape = medoids.shape
	if len(medoids_shape) ==1:
		medoids = medoids.reshape((1,len(medoids)))
	k = len(medoids)
	S = np.empty((m,k))
	for i in range(m):
		d_i = np.linalg.norm(X[i,:] -  medoids,ord=p,axis=1)
		S[i,:] = d_i**p

	return S

def assign_labels(S):
	return np.argmin(S,axis=1)

def update_medoids(points,medoids,p):
	S = compute_d_p(points,medoids,p)
	labels = assign_labels(S)
	out_medoids = medoids

	for i in set(labels):
		avg_dissimilarity = np.sum(compute_d_p(points,medoids[i],p))
		cluster_points = points[labels==i]
		print("AD ",avg_dissimilarity)

		for datap in cluster_points:
			new_medoid = datap
			new_dissimilarity = np.sum(compute_d_p(points,datap,p))
			print("DD ",datap)

			if new_dissimilarity < avg_dissimilarity:
				print("UD ",datap)
				avg_dissimilarity = new_dissimilarity
				out_medoids[i] = datap
				print("ND ",new_dissimilarity)

	return out_medoids

def has_converged(old_medoids,medoids):
	return set([tuple(x) for x in old_medoids]) == set([tuple(x) for x in medoids])

def kmedoids(X,k,p,starting_medoids=None,max_steps=np.inf):
	if starting_medoids is None:
		medoids = init_medoids(X,k)
	else:
		medoids = starting_medoids

	print("M ",medoids)
	converged = False
	labels = np.zeros(len(X))
	i = 1
	while (not converged) and (i<=max_steps):
		old_medoids = medoids.copy()
		S = compute_d_p(X,medoids,p)
		print("S ",S)
		labels = assign_labels(S)
		print("LAB ",labels)
		medoids = update_medoids(X,medoids,p)
		print("NEW M ",medoids)
		converged = has_converged(old_medoids,medoids)
		i+=1
	return(medoids,labels)

def main():
	X = np.array([(5.4,5),(4,4.5),(3.7,5),(3.6,4.3),(3,4),(4.7,5),(3,5),(4,4),(4.3,5.3),(5,5.5)])
	# print(X)
	medoids,labels = kmedoids(X,2,2,max_steps=1000)
	print(medoids)
	print("___________________________________",labels)

if __name__ == "__main__":
	main()