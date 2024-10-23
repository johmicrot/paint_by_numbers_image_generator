# color_reduction.py

import numpy as np
from sklearn.cluster import KMeans

def reduce_colors_kmeans(image, color_count):
    """Reduces the number of colors in the image using KMeans clustering."""
    color_array = image.reshape(-1, 3)
    kmeans = KMeans(n_clusters=color_count, random_state=42).fit(color_array)
    kmeans_labels = kmeans.labels_
    centers = kmeans.cluster_centers_.astype('uint8')
    # Map each pixel to its cluster center
    kmeans_image = centers[kmeans_labels].reshape(image.shape)
    return kmeans_image, centers, kmeans_labels
