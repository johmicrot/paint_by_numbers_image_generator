# segmentation.py

import numpy as np
import os
import matplotlib.pyplot as plt
from skimage import color, graph
import logging
def _weight_mean_color(graph, src, dst, n):
    """Callback to handle merging nodes by recomputing mean color.

    The method expects that the mean color of `dst` is already computed.

    Parameters
    ----------
    graph : RAG
        The graph under consideration.
    src, dst : int
        The vertices in `graph` to be merged.
    n : int
        A neighbor of `src` or `dst` or both.

    Returns
    -------
    data : dict
        A dictionary with the `"weight"` attribute set as the absolute
        difference of the mean color between node `dst` and `n`.
    """

    diff = graph.nodes[dst]['mean color'] - graph.nodes[n]['mean color']
    diff = np.linalg.norm(diff)
    return {'weight': diff}


def merge_mean_color(graph, src, dst):
    """Callback called before merging two nodes of a mean color distance graph.

    This method computes the mean color of `dst`.

    Parameters
    ----------
    graph : RAG
        The graph under consideration.
    src, dst : int
        The vertices in `graph` to be merged.
    """
    graph.nodes[dst]['total color'] += graph.nodes[src]['total color']
    graph.nodes[dst]['pixel count'] += graph.nodes[src]['pixel count']
    graph.nodes[dst]['mean color'] = (
        graph.nodes[dst]['total color'] / graph.nodes[dst]['pixel count']
    )

def perform_rag_merging(image, slic_labeled_image, slic_to_kmeans, output_folder):
    from collections import defaultdict

    thresh = 5
    g = graph.rag_mean_color(image, slic_labeled_image)
    labels2 = graph.merge_hierarchical(
        slic_labeled_image,
        g,
        thresh=thresh,
        rag_copy=False,
        in_place_merge=True,
        merge_func=merge_mean_color,
        weight_func=_weight_mean_color,
    )

    out = color.label2rgb(labels2, image, kind='avg', bg_label=0)
    plt.imsave( os.path.join(output_folder, 'reduced_color_image.png'), out)
    return labels2  # Return the NumPy array directly
