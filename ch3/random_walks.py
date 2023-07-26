import networkx as nx
import matplotlib.pyplot as plt
import numpy as np
import random
random.seed(0)
from gensim.models import Word2Vec
from sklearn.manifold import TSNE
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

G = nx.erdos_renyi_graph(10, 0.3, seed=1, directed=False)

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


start = 0
length = 10

def random_walk(start, length):
    walk = [str(start)]
    for i in range(length):
        neighbors = [node for node in G.neighbors(start)]
        next_node = np.random.choice(neighbors, 1)[0]
        walk.append(str(next_node))
        start = next_node
    return walk    


random_walk(1, 5)

draw_g(G)

# |> KARATE -----------------------------------------------------------

G = nx.karate_club_graph()

# Each node has a dictionary of features. 
G.nodes[0]

labels = []
for node in G.nodes:
    label = G.nodes[node]["club"]
    # print(label)
    labels.append(1 if label == 'Officer' else 0)

set(labels)

plt.figure(figsize=(12,12), dpi=300)
plt.axis('off')
nx.draw_networkx(G,
                 pos=nx.spring_layout(G, seed=0),
                 node_color=labels,
                 node_size=800,
                 cmap='coolwarm',
                 font_size=14,
                 font_color='white'
                 )

# Let's make our training data
walks = []
len(G.nodes) * 80
for node in G.nodes:
    for _ in range(80):
        walks.append(random_walk(node, 10))

np.array(walks).shape
walks[0]

model = Word2Vec(walks,
                 hs=1,   # Hierarchical softmax
                 sg=1,   # Skip-gram
                 vector_size=100,
                 window=10,
                 workers=2,
                 seed=0)


model.train(walks, total_examples=model.corpus_count, epochs=20, report_delay=1)


print('Nodes that are the most similar to node 0:')
node = 0
for similarity in model.wv.most_similar(positive=[str(node)]):
    print(f'   {similarity}')


# Similarity between two nodes
print(f"Similarity between node 0 and 4: {model.wv.similarity('0', '4')}")


nodes_wv = np.array([model.wv.get_vector(str(i)) for i in range(len(model.wv))])
nodes_wv.shape
labels = np.array(labels)

tsne = TSNE(n_components=2,
            learning_rate='auto',
            init='pca',
            random_state=0).fit_transform(nodes_wv)

plt.figure(figsize=(6, 6), dpi=300)
plt.scatter(tsne[:, 0], tsne[:, 1], s=100, c=labels, cmap="coolwarm")
plt.show()

train_mask = [0, 2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22, 24, 26, 28]
test_mask = [1, 3, 5, 7, 9, 11, 13, 15, 17, 19, 21, 23, 25, 27, 29, 30, 31, 32, 33]

clf = RandomForestClassifier(random_state=0)
clf.fit(nodes_wv[train_mask], labels[train_mask])

y_pred = clf.predict(nodes_wv[test_mask])
a = accuracy_score(y_pred, labels[test_mask])

type(a)
a


























