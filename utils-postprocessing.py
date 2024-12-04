import random

def enforce_fixed_cluster_size(labels, file_paths, images, fixed_size=4):
    """
    Ensure each cluster contains exactly 'fixed_size' images.
    """
    print("Enforcing fixed cluster size...")
    unique_labels = set(labels)
    new_labels = []
    new_images = []
    new_file_paths = []
    
    for label in unique_labels:
        indices = [i for i, lbl in enumerate(labels) if lbl == label]
        if len(indices) >= fixed_size:
            selected_indices = random.sample(indices, fixed_size)
        else:
            print(f"Cluster {label} has fewer than {fixed_size} images. Keeping all.")
            selected_indices = indices
        new_labels.extend([label] * len(selected_indices))
        new_images.extend([images[i] for i in selected_indices])
        new_file_paths.extend([file_paths[i] for i in selected_indices])
    
    return new_labels, new_file_paths, new_images
