from sklearn.cluster import KMeans
from sklearn.preprocessing import normalize
from sklearn.decomposition import PCA
from .base_clustering import ClusteringMethod

class KMeansClustering(ClusteringMethod):
    """
    Implementation of K-Means clustering.
    """
    def __init__(self, n_clusters):
        self.n_clusters = n_clusters

    def cluster(self, embeddings):
        """
        Perform K-Means clustering on the embeddings.
        """
        print("Performing K-Means clustering...")
        embeddings = normalize(embeddings)  # Normalize embeddings
        embeddings = PCA(n_components=40).fit_transform(embeddings)  # Dimensionality reduction
        kmeans = KMeans(n_clusters=self.n_clusters, random_state=42)
        labels = kmeans.fit_predict(embeddings)
        return labels
