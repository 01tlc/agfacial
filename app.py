from models.facenet_model import FaceNetModel
from clustering.kmeans_clustering import KMeansClustering
from utils.image_utils import load_and_preprocess_images
from utils.visualization import visualize_clusters, show_cluster_images
from utils.postprocessing import enforce_fixed_cluster_size

class FaceClusteringApp:
    """
    Main application for face clustering.
    """
    def __init__(self, dataset_path, model, clustering_method):
        self.dataset_path = dataset_path
        self.model = model
        self.clustering_method = clustering_method

    def run(self, fixed_size=4):
        # Load and preprocess images
        images, file_paths = load_and_preprocess_images(self.dataset_path)

        # Load model
        self.model.load_model()

        # Extract embeddings
        embeddings = self.model.extract_embeddings(images)

        # Perform clustering
        labels = self.clustering_method.cluster(embeddings)

        # Enforce fixed cluster size
        labels, file_paths, images = enforce_fixed_cluster_size(labels, file_paths, images, fixed_size=fixed_size)

        # Visualize results
        visualize_clusters(labels, file_paths)
        show_cluster_images(labels, images)


if __name__ == "__main__":
    dataset_path = "./dataset_faces"
    model = FaceNetModel()
    clustering_method = KMeansClustering(n_clusters=10)
    app = FaceClusteringApp(dataset_path, model, clustering_method)
    app.run(fixed_size=4)
