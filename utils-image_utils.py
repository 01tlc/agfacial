import os
import cv2
import numpy as np

def load_and_preprocess_images(dataset_path):
    """
    Load and preprocess images from the dataset.
    """
    print("Loading and preprocessing images...")
    if not os.path.exists(dataset_path):
        raise FileNotFoundError(f"Dataset path not found: {dataset_path}")

    images = []
    file_paths = []
    for root, _, files in os.walk(dataset_path):
        for file in files:
            if file.lower().endswith((".jpg", ".png")):
                file_path = os.path.join(root, file)
                image = cv2.imread(file_path)
                if image is None:
                    print(f"Failed to load image: {file_path}")
                    continue
                image = cv2.resize(image, (160, 160))  # FaceNet input size
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB) / 255.0  # Normalize to [0, 1]
                images.append(image)
                file_paths.append(file_path)
    print(f"{len(images)} images loaded.")
    return np.array(images), file_paths
