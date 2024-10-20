# Aim: To Implement Graph Based Clustering.

import numpy as np
import matplotlib.pyplot as plt #pip install matplotlib
import seaborn as sns #pip install seaborn
import warnings
from sklearn.datasets import make_blobs #pip install scikit-learn
from mst_clustering import MSTClustering #pip install mst_clustering

# Ignore specific warnings
warnings.filterwarnings("ignore", message="elementwise")

# Define a function to plot the MST clustering results
def plot_mst(model, cmap='rainbow'):
    """
    Plot the Minimum Spanning Tree (MST) clustering results.
    
    Parameters:
    model: MSTClustering object that has been fit to the data.
    cmap: Color map for visualizing clusters.
    """
    # Extract the data points from the fitted model
    X = model.X_fit_
    
    # Create a figure with 2 subplots for full and trimmed MSTs
    fig, ax = plt.subplots(1, 2, figsize=(16, 6), sharex=True, sharey=True)
    
    # Plot the full MST and the trimmed MST side by side
    for axi, full_graph, colors in zip(ax, [True, False], ['lightblue', model.labels_]):
        # Get the graph segments (edges) from the MST model
        segments = model.get_graph_segments(full_graph=full_graph)
        
        # Plot the MST edges
        axi.plot(segments[0], segments[1], '-k', zorder=1, lw=1)
        
        # Plot the data points with colors corresponding to their cluster labels
        scatter = axi.scatter(X[:, 0], X[:, 1], c=colors, cmap=cmap, zorder=2)
        axi.axis('tight')
    
    # Set the titles for the subplots
    ax[0].set_title('Full Minimum Spanning Tree', size=16)
    ax[1].set_title('Trimmed Minimum Spanning Tree', size=16)
    
    # Add a color bar to show cluster labels
    fig.colorbar(scatter, ax=ax, orientation='vertical', fraction=0.02, pad=0.04)

# Generate synthetic data with 200 samples and 6 centers
X, y = make_blobs(200, centers=6)

# Plot the generated synthetic data points
plt.figure(figsize=(8, 6))
plt.scatter(X[:, 0], X[:, 1], c='lightblue')
plt.title('Generated Data', size=16)
plt.show()

# Initialize and fit the MSTClustering model
model = MSTClustering(cutoff_scale=2, approximate=False)

# Predict the cluster labels using the fitted MST model
labels = model.fit_predict(X)

# Plot the clustered data points using the predicted labels
plt.figure(figsize=(8, 6))
plt.scatter(X[:, 0], X[:, 1], c=labels, cmap='rainbow')
plt.title('MST Clustering', size=16)
plt.show()

# Plot the MST with the clustering results
plot_mst(model)
