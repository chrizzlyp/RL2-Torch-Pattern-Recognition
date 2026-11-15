"""K-Means specific code for the hidden state analysis

Usage in a notebook inside experiments/analysis/kmeans/<system>/:

    sys.path.insert(0, os.path.abspath(".."))
    from kmeans_utils import fit_kmeans, prediction_strength
"""

import numpy as np
from sklearn.cluster import KMeans

# Colors per cluster
CLUSTER_COLORS = [
    "#2a78d6",
    "#eb6834",
    "#1baf7a",
    "#eda100",
    "#e87ba4",
    "#008300",
    "#6250d6",
    "#e34948",
]


def fit_kmeans(hidden_states, k):
    """K-Means with the settings of all notebooks: 100 initializations with best solution kept."""
    return KMeans(n_clusters=k, random_state=42, n_init=100).fit(hidden_states)


def sort_clusters_by_task_parameter(labels, parameters, k):
    """Sort the clusters by their mean task parameter and number them in this order.

    Cluster 1 has the most negative mean task parameter, cluster k the most positive.
    Returns the mean task parameter of every cluster, the clusters sorted by it,
    and the number of every cluster.
    """
    cluster_means = np.array(
        [parameters[labels == cluster].mean() for cluster in range(k)]
    )
    cluster_order = np.argsort(cluster_means)
    cluster_number = np.empty(k, dtype=int)
    cluster_number[cluster_order] = np.arange(1, k + 1)
    return cluster_means, cluster_order, cluster_number


def prediction_strength(reference_kmeans, test_hidden_states, k):
    """Prediction strength of Tibshirani & Walther (2005) for one test sample.

    The test sample is clustered on its own. For every test cluster, the share of task pairs
    that the reference centroids also assign to the same cluster is calculated.
    The weakest cluster determines the prediction strength.
    """
    test_assignments = fit_kmeans(test_hidden_states, k).labels_
    reference_assignments = reference_kmeans.predict(test_hidden_states)

    cluster_strengths = []

    for cluster in range(k):
        members = np.flatnonzero(test_assignments == cluster)
        n = len(members)

        # A cluster with a single task contains no pairs (counts as 1, as in the R package fpc)
        if n < 2:
            cluster_strengths.append(1.0)
            continue

        same_centroid = (
            reference_assignments[members][:, None]
            == reference_assignments[members][None, :]
        )
        cluster_strengths.append((same_centroid.sum() - n) / (n * (n - 1)))

    return min(cluster_strengths)
