# importing modules 
import networkx as nx 
import matplotlib.pyplot as plt 

G = nx.DiGarph() 

G.add_edges_from([('A', 'D'), ('B', 'C'), ('B', 'E'), ('C', 'A'), 
				('D', 'C'), ('E', 'D'), ('E', 'B'), ('E', 'F'), 
				('E', 'C'), ('F', 'C'), ('F', 'H'), ('G', 'A'), 
				('G', 'C'), ('H', 'A')]) 

plt.figure(figsize =(10, 10)) 
nx.draw_networkx(G, with_labels = True) 

hubs, authorities = nx.hits(G, max_iter = 50, normalized = True) 
# The in-built hits function returns two dictionaries keyed by nodes 
# containing hub scores and authority scores respectively. 

print("Hub Scores: ", hubs) 
print("Authority Scores: ", authorities) 
