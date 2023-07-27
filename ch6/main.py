import networkx as nx
import random
random.seed(0)
import numpy as np
np.random.seed(0)
from src.utils import draw_g

G = nx.erdos_renyi_graph(10, 0.3, seed=1, directed=False)

draw_g(G)
G.nodes


previous = 0
current = 8
p=1
q=1

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


import numpy as np
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt

# Define the model
def exponential_decline(t, q0, b):
  return q0 * np.exp(-b * t)

# Create the data
t = np.linspace(0, 100, 1000)
q = exponential_decline(t, 100, 0.05)

# Add noise to the data
noise = np.random.normal(0, 5, len(q))
q += noise

# Introduce gaps in the data
gap_probability = 0.1
for i in range(len(q)):
  if random.random() < gap_probability:
    q[i] = np.nan

# Simulate a production decline
for i in range(len(q)):
  if random.random() < 0.5:
    q[i] -= random.uniform(0, 10)

# Plot the data
plt.plot(t, q)
plt.show()

# Fit the model to the data
popt, _ = curve_fit(exponential_decline, t, q)

# Get the optimal value of -b
b = -popt[1]

# Print the optimal value of -b
print('The optimal value of -b is:', b)

import random

# Create the data
t = np.linspace(0, 100, 1000)
q = exponential_decline(t, 100, 0.05)

# Add noise to the data
noise = np.random.normal(0, 5 / len(q), len(q))
q += noise

# Calculate the drop size
drop_size = random.uniform(0, 100)

# Define the event probability
event_probability = 0.1

# Initialize the duration variable
duration = int(random.uniform(10, 100))

# Add the random event
for i in range(len(q)):
  if random.random() < event_probability:
    q[i] -= drop_size
    for j in range(i, i+duration):
      q[j] = 0

# Plot the data
plt.plot(t, q)
plt.show()


