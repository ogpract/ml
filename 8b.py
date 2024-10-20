import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.cluster import SpectralClustering
from sklearn.metrics.pairwise import pairwise_distances
import networkx as nx
# Generate synthetic data
X, y = datasets.make_classification(n_samples=100, n_features=2, n_informative=2, n_redundant=0,
                                    n_clusters_per_class=1, random_state=42)
# Apply Spectral Clustering
# Number of clusters based on the number of classes
n_clusters = len(np.unique(y))
spectral = SpectralClustering(
    n_clusters=n_clusters, affinity='nearest_neighbors', random_state=42)
labels = spectral.fit_predict(X)


def plot_clusters(X, labels):
    plt.figure(figsize=(10, 6))
    scatter = plt.scatter(X[:, 0], X[:, 1], c=labels,
                          edgecolors='k', cmap=plt.cm.RdYlBu)
    plt.title("Clusters with Spectral Clustering")
    plt.xlabel("Feature 1")
    plt.ylabel("Feature 2")
    plt.colorbar(scatter)
    plt.show()


plot_clusters(X, labels)
# Create a graph from pairwise distances
distances = pairwise_distances(X)
G = nx.Graph()
# Add edges based on distance
for i in range(len(X)):
    for j in range(i + 1, len(X)):
        G.add_edge(i, j, weight=distances[i, j])
# Compute the MST
mst = nx.minimum_spanning_tree(G)
# Convert MST to a simple graph for visualization
mst = nx.Graph(mst)


def plot_mst(X, mst):
    pos = {i: (X[i, 0], X[i, 1]) for i in range(len(X))}
    plt.figure(figsize=(10, 6))
    nx.draw(mst, pos, with_labels=True, node_color='lightblue', edge_color='gray', node_size=100,
            font_size=12, font_weight='bold')
    plt.title("Minimum Spanning Tree")
    plt.xlabel("Feature 1")
    plt.ylabel("Feature 2")
    plt.show()


plot_mst(X, mst)
