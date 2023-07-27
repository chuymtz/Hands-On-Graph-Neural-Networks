import matplotlib.pyplot as plt
import networkx as nx

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