import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import load_iris
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import AgglomerativeClustering
from scipy.cluster.hierarchy import dendrogram, linkage

# Load the Iris dataset
iris = load_iris()
X = iris.data
y = iris.target

# Convert to DataFrame for better visualization
iris_df = pd.DataFrame(X, columns=iris.feature_names)

# Standardize the features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Perform Agglomerative Hierarchical Clustering
agglo = AgglomerativeClustering(n_clusters=3)
clusters = agglo.fit_predict(X_scaled)

# Add the cluster labels to the DataFrame
iris_df['Cluster'] = clusters

# Visualize the clusters
plt.figure(figsize=(10, 6))
sns.scatterplot(data=iris_df, x='sepal length (cm)', y='sepal width (cm)', hue='Cluster', palette='Set1', s=100)
plt.title('Agglomerative Hierarchical Clustering on Iris Dataset')
plt.xlabel('Sepal Length (cm)')
plt.ylabel('Sepal Width (cm)')
plt.legend(title='Cluster')
plt.show()

# Create a dendrogram to visualize the hierarchical clustering
linkage_matrix = linkage(X_scaled, method='ward')
plt.figure(figsize=(12, 6))
dendrogram(linkage_matrix, labels=iris.target_names[y], leaf_rotation=90)
plt.title('Dendrogram for Agglomerative Hierarchical Clustering')
plt.xlabel('Iris Species')
plt.ylabel('Distance')
plt.show()