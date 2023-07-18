import torch

import networkx as nx

G = nx.Graph()
G.add_edges_from([('A', 'B'), ('A', 'C'), ('B', 'D'), ('B', 'E'), ('C', 'F'), ('C', 'G')])



#  BREATH FIRST SEARCH

graph = G
node = "A"

def bfs(graph, node):
    visited, queue = [node], [node]

    while queue: # continues until queue is empty
        print(1)
        node = queue.pop(0)
        
        for neighbor in graph[node]:
            if neighbor not in visited:
                visited.append(neighbor)
                queue.append(neighbor)
    return visited
    

v = bfs(G, "A")

nx.draw(G,  with_labels=True,)

 
    
    
    
    
    

    
    









