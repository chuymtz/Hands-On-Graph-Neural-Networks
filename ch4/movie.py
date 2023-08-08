from pprint import pprint
import networkx as nx
from io import BytesIO
from urllib.request import urlopen
from zipfile import ZipFile
import pandas as pd
from collections import defaultdict
import node2vec

url = 'https://files.grouplens.org/datasets/movielens/ml-100k.zip'
with urlopen(url) as zurl:
    with ZipFile(BytesIO(zurl.read())) as zfile:
        zfile.extractall('.')
        
ratings = pd.read_csv('ml-100k/u.data', sep='\t', names=['user_id', 'movie_id', 'rating', 'unix_timestamp'])
ratings

movies = pd.read_csv('ml-100k/u.item', sep='|', usecols=range(2), names=['movie_id', 'title'], encoding='latin-1')        

ratings = ratings[ratings.rating >= 4].reset_index(drop=True)
pprint(ratings)

pairs = defaultdict(int)
for group in ratings.groupby("user_id"):
    type(group)
    # group[0] # user id
    # group[1] # groupe tabl
    user_movies = list(group[1]["movie_id"])
    for i in range(len(user_movies)):
        for j in range(i+1, len(user_movies)):
            pairs[user_movies[i], user_movies[j]] += 1
            

G = nx.Graph()

for pair in pairs:
    movie1, movie2 = pair
    score = pairs[pair]
    if score >= 20:
        G.add_edge(movie1, movie2, weight=score)

nx.draw(G)

from node2vec import node2vec

node2vec = node2vec(G, dimensions=64, walk_length=20, num_walks=200, p=2, q=1, workers=1)
model = node2vec.fit(window=10, min_count=1, batch_words=4)


import networkx as nx
from node2vec import Node2Vec

# Create a graph
graph = nx.fast_gnp_random_graph(n=100, p=0.5)

# Precompute probabilities and generate walks - **ON WINDOWS ONLY WORKS WITH workers=1**
node2vec = Node2Vec(graph, dimensions=64, walk_length=30, num_walks=200, workers=4)