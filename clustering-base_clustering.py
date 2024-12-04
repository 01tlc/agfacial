class ClusteringMethod:
    """
    Abstract base class for clustering methods.
    """
    def cluster(self, embeddings):
        """
        Perform clustering on the embeddings.
        """
        raise NotImplementedError
