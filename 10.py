# Aim: To implement Principal Component Analysis (PCA). 

import numpy as np 
import matplotlib.pyplot as plt 
from sklearn.decomposition import PCA 
from sklearn.preprocessing import StandardScaler 
from sklearn.datasets import load_iris 

# Load dataset 
data = load_iris() 
X = data.data 
y = data.target 
feature_names = data.feature_names 

# Standardize the data 
scaler = StandardScaler() 
X_scaled = scaler.fit_transform(X) 

# Apply PCA 
pca = PCA(n_components=4) # PCA with all components 
X_pca = pca.fit_transform(X_scaled) 

# Plot 2D projection 
plt.figure(figsize=(12, 5)) 

# Scatter plot of first two principal components 
plt.subplot(1, 2, 1) 
scatter = plt.scatter(X_pca[:, 0], X_pca[:, 1], c=y, cmap='viridis') 
plt.colorbar(scatter, label='Target') 
plt.xlabel('Principal Component 1') 
plt.ylabel('Principal Component 2') 
plt.title('2D PCA Projection') 

# Bar chart of explained variance ratio 
plt.subplot(1, 2, 2) 
explained_variance_ratio = pca.explained_variance_ratio_ 
plt.bar(range(1, len(explained_variance_ratio) + 1), explained_variance_ratio, alpha=0.7) 
plt.xlabel('Principal Component') 
plt.ylabel('Explained Variance Ratio') 
plt.title('Explained Variance Ratio') 

plt.tight_layout() 
plt.show()
