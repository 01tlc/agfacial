import matplotlib.pyplot as plt

def visualize_clusters(labels, file_paths):
    """
    Print the cluster file paths.
    """
    unique_labels = set(labels)
    for label in unique_labels:
        print(f"Cluster {label}:")
        cluster_images = [file_paths[i] for i in range(len(labels)) if labels[i] == label]
        for img_path in cluster_images:
            print(f"  - {img_path}")

def show_cluster_images(labels, images):
    """
    Visually display images grouped by clusters.
    """
    unique_labels = set(labels)
    for label in unique_labels:
        cluster_images = [images[i] for i in range(len(labels)) if labels[i] == label]
        plt.figure(figsize=(10, 5))
        for i, img in enumerate(cluster_images[:10]):  # Show max 10 images per cluster
            plt.subplot(2, 5, i + 1)
            plt.imshow(img)
            plt.axis("off")
        plt.suptitle(f"Cluster {label}")
        plt.show()
