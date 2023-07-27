import matplotlib.pyplot as plt
import networkx as nx
import numpy as np

def draw_g(G):
    plt.figure(dpi=300)
    plt.axis('off')
    nx.draw_networkx(G,
        pos=nx.spring_layout(G, seed=0),
        node_size=600,
        cmap='coolwarm',
        # font_size=14,
        # font_color='white'
        )
    
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