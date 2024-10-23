# segmentation.py

import numpy as np
import matplotlib.pyplot as plt
import os
from skimage import segmentation, color
from skimage import graph

def perform_slic_segmentation(image_list, compactness, segment_count, sigma):
    """Performs SLIC segmentation on a list of images."""
    slic_label_images = []
    for img in image_list:
        slic_label_images.append(segmentation.slic(
            img,
            compactness=compactness,
            n_segments=segment_count,
            start_label=1,
            sigma=sigma
        ))
    return slic_label_images

def perform_rag_merging(image_list, slic_label_images, color_threshold, debug_folder):
    """
    Merges regions in the segmented images using RAG with exact color matching.  This code might not be needed

    Parameters:
    - image_list: List of images.
    - slic_label_images: List of SLIC label images.
    - color_threshold: Threshold for region merging in the RAG (0.0 for exact color matching).
    - debug_folder: Path to save debug images.

    Returns:
    - rag_label_images: List of label images after merging.
    """
    rag_label_images = []
    for index, slic_label_image in enumerate(slic_label_images):
        # Construct RAG from segmented image using mean color similarity
        rag = graph.rag_mean_color(image_list[index], slic_label_image, mode='distance')

        # Save the label image before merging
        label_image_before = color.label2rgb(slic_label_image, image_list[index], kind='avg', bg_label=-1)
        label_image_before = (label_image_before ).astype(np.uint8)
        before_merge_path = os.path.join(debug_folder, f'label_image_before_merge_{index}.png')
        plt.imsave(before_merge_path, label_image_before)

        # Define custom merging criteria: exact color match
        labels = graph.merge_hierarchical(
            slic_label_image,
            rag,
            thresh=color_threshold,
            rag_copy=False,
            in_place_merge=True,
            merge_func=lambda rag, src, dst: None,  # No additional action needed on merge
            weight_func=lambda rag, src, dst, n: exact_color_weight(rag, dst, n)
        )
        rag_label_images.append(labels)

        # Save the label image after merging
        label_image_after = color.label2rgb(labels, image_list[index], kind='avg', bg_label=-1)
        label_image_after = (label_image_after).astype(np.uint8)
        after_merge_path = os.path.join(debug_folder, f'label_image_after_merge_{index}.png')
        plt.imsave(after_merge_path, label_image_after)

    return rag_label_images

def exact_color_weight(rag, dst, n):
    """
    Determines the weight based on exact color matching.

    Parameters:
    - rag: The Region Adjacency Graph.
    - dst: Destination node.
    - n: Neighbor node.

    Returns:
    - A dictionary with the 'weight' key.
    """
    # Retrieve mean colors of the destination and neighbor nodes
    mean_color_dst = rag.nodes[dst]['mean color']
    mean_color_n = rag.nodes[n]['mean color']

    # Check for exact color match
    if np.array_equal(mean_color_dst, mean_color_n):
        return {'weight': 0}  # Allow merging
    else:
        return {'weight': float('inf')}  # Prevent merging
