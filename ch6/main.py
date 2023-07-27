import networkx as nx
import random
random.seed(0)
import numpy as np
np.random.seed(0)
from src.utils import draw_g

G = nx.erdos_renyi_graph(10, 0.3, seed=1, directed=False)

draw_g(G)
G.nodes


# previous = 0
# current = 8
# p=1
# q=1

# How to get the 
def next_node(G, previous, current, p, q):
    neighbors = list(G.neighbors(current))
    
    # pi_{i, j} is the transition prob of moving to next neighbor. 
    # this is decompsed as pi = alpha  * omega
    # omega is the weight of the edge
    # alpha is known as the search bias
    # we also need d_{a, b} which si shortest path distance between a and p
    alphas = []
    for neighbor in neighbors:
        if neighbor == previous:
            alpha = 1/p
        elif G.has_edge(neighbor, previous):
            alpha = 1
        else:
            alpha = 1/q
        
        alphas.append(alpha)
    
    
    probs = [alpha / sum(alphas) for alpha in alphas]

    next = np.random.choice(neighbors, size=1, p=probs)[0]
    return next

def random_walk(G, start, length, p, q):
    walk = [start]
    for i in range(length):
        current = walk[-1]
        previous = walk[-2] if len(walk) > 1 else None
        next = next_node(G, previous, current, p, q)
        walk.append(next)
    return [str(x) for x in walk]

# next_ = next_node(G, 4, 3, 1, 1)
random_walk(G, 0, 8, p=1, q=1)
random_walk(G, 0, 8, p=1, q=10)
random_walk(G, 0, 8, p=10, q=1)
draw_g(G)
















