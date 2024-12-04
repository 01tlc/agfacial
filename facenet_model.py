from keras_facenet import FaceNet
import numpy as np
from .base_model import EmbeddingModel

class FaceNetModel(EmbeddingModel):
    """
    Implementation of the FaceNet embedding model.
    """
    def __init__(self):
        self.model = None

    def load_model(self):
        print("Loading FaceNet model...")
        self.model = FaceNet().model
        print("FaceNet model loaded successfully.")

    def extract_embeddings(self, images):
        """
        Extract embeddings for a batch of images.
        """
        print("Extracting embeddings using FaceNet...")
        embeddings = []
        for i, image in enumerate(images):
            try:
                embedding = self.model.predict(np.expand_dims(image, axis=0))[0]
                embeddings.append(embedding)
            except Exception as e:
                print(f"Error extracting embedding for image {i}: {e}")
        return np.array(embeddings)
