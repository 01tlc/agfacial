class EmbeddingModel:
    """
    Abstract base class for embedding models.
    """
    def load_model(self):
        """
        Load the embedding model.
        """
        raise NotImplementedError

    def extract_embeddings(self, images):
        """
        Extract embeddings from a batch of images.
        """
        raise NotImplementedError
