#%%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
from sklearn.decomposition import PCA

data = pd.read_csv("/Users/francis1223/Documents/Robotics MSc TB1/AI/C/dungeon_sensorstats_partC.csv")
data.info()
print(data["race"].unique()) 
print(data.isna().sum()) 

data = data.drop(columns = ["race", "bribe"]) 
data = data.dropna() 
print(data.isna().sum()) 

encoder = LabelEncoder()
data["alignment"] = encoder.fit_transform(data["alignment"])
print(data)
data.info()

pca = PCA(n_components=2)
data_pca = pca.fit_transform(data)
plt.scatter(data_pca[:, 0], data_pca[:, 1], s=10)
plt.legend(fontsize=14)
plt.xlabel("X")
plt.ylabel("Y")
plt.title("PCA Projection 9D to 2D")
plt.show()
#%%
from sklearn.mixture import GaussianMixture
from sklearn.mixture import GaussianMixture as GMM

# Generate a plot showing the aic and bic scores
bics = []
aics = []
K = 10
for k in range(1, K+1):
    gmm = GMM(n_components=k, random_state=42)
    gmm.fit(data_pca)

    bics.append(gmm.bic(data_pca))
    aics.append(gmm.aic(data_pca))

plt.figure(figsize=(6, 10))
plt.plot(range(1, K+1), bics, 'bx-', label='BIC')
plt.plot(range(1, K+1), aics, 'r.--', label='AIC')
plt.xlabel('k', fontsize=14)
plt.ylabel('Information Criterion', fontsize=14)
plt.legend(fontsize=14)
plt.show()

optimal_k_bic = np.arange(1, K+1)[np.argmin(bics)]
optimal_k_aic = np.arange(1, K+1)[np.argmin(aics)]
print(f"Optimak k (bic): {optimal_k_bic}")
print(f"Optimak k (aic): {optimal_k_aic}")


# %%
gmm = GaussianMixture(n_components=optimal_k_aic, random_state=42)
gmm.fit(data_pca)
gmm_labels = gmm.predict(data_pca)

plt.scatter(data_pca[:, 0], data_pca[:, 1], c=gmm_labels, s=10)
plt.scatter(gmm.means_[:,0], gmm.means_[:,1], c = 'red', s = 50, marker='*', label = 'Centroids')
plt.legend(fontsize=14)
plt.xlabel('X', fontsize=14)
plt.ylabel('Y', fontsize=14)
plt.autoscale()
plt.show()


# %%
# KNN baseline
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from scipy.spatial import Voronoi, voronoi_plot_2d

# Elbow
inertias = []
K=10
for k in range(1, K+1):

    kmeans = KMeans(n_clusters=k, random_state=42)
    kmeans.fit(data_pca)
    calculate_inertia = kmeans.inertia_
    inertias.append(calculate_inertia)

plt.figure(figsize=(6, 10))
plt.plot(range(1, K+1), inertias)
plt.xlabel('k', fontsize=14)
plt.ylabel('Inertia', fontsize=14)
plt.legend(fontsize=14)
plt.show()

# Silhouette_scores
silhouette_scores = []
for k in range(2, K+1):
  kmeans = KMeans(n_clusters=k, random_state=42)
  kmeans.fit(data_pca)
  silhouette_scores.append(silhouette_score(data_pca, kmeans.labels_))

plt.figure(figsize=(6, 10))
plt.plot(range(2, K+1), silhouette_scores)
plt.xlabel('k', fontsize=14)
plt.ylabel('Silhouette score', fontsize=14)
plt.legend(fontsize=14)
plt.show()

optimal_k = np.arange(2, K+1)[np.argmax(silhouette_scores)]
print(f"Optimal_k: {optimal_k}")
# %%
kmeans = KMeans(n_clusters=optimal_k, random_state=42)
kmeans.fit(data_pca)


plt.scatter(data_pca[:, 0], data_pca[:, 1], c=kmeans.labels_, s=10)
plt.scatter(kmeans.cluster_centers_[:,0], kmeans.cluster_centers_[:,1], c = 'red', s = 50, marker='*', label = 'Centroids')
plt.xlabel('X', fontsize=14)
plt.ylabel('Y', fontsize=14)
plt.legend(fontsize=14)
plt.autoscale()
plt.show()


# %%
