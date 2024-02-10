from sklearn import cluster as sk_cluster
from sklearn.neighbors import kneighbors_graph
from sklearn.preprocessing import StandardScaler
import numpy as np


def get_clusters(sources, graph_ax, min_size, max_size) -> dict:
    # Loop Through Every Modal
    params = {
        # "quantile": 0.3,
        # "eps": 0.18,
        # "damping": 0.7,
        # "preference": -200,
        "n_neighbors": 2,
        # "min_samples": 7,
        # "xi": 0.01,
        # "min_cluster_size": None,
        # "allow_single_cluster": False,
        # "hdbscan_min_cluster_size": 2,
        # "hdbscan_min_samples": 3,
        "random_state": 42,
        "n_clusters": max(len(sources) // 40, 1),
    }

    # normalize dataset for easier parameter selection
    X = StandardScaler().fit_transform(sources)

    # connectivity matrix for structured Ward
    connectivity = kneighbors_graph(
        X, n_neighbors=params["n_neighbors"], include_self=False
    )
    # make connectivity symmetric
    connectivity = 0.5 * (connectivity + connectivity.T)
    modals = [
        ("MiniBatchKMeans",
         sk_cluster.MiniBatchKMeans(
             n_clusters=params["n_clusters"],
             n_init="auto",
             random_state=params["random_state"],
         )),
        ("Ward",
         sk_cluster.AgglomerativeClustering(
             n_clusters=params["n_clusters"], linkage="ward", connectivity=connectivity
         )),
        ("Agglomerative Clustering",
         sk_cluster.AgglomerativeClustering(
             linkage="average",
             metric="cityblock",
             n_clusters=params["n_clusters"],
             connectivity=connectivity,
         )),
        ("Birch",
         sk_cluster.Birch(n_clusters=params["n_clusters"])),
    ]
    cluster_graphs = {}
    for (name, modal) in modals:
        # model = modal(damping=0.9)
        modal.fit(sources)
        if hasattr(modal, "labels_"):
            yhat = modal.labels_.astype(int)
        else:
            yhat = modal.predict(sources)
        clusters = np.unique(yhat)
        for n, cluster in enumerate(clusters):
            row_ix = np.where(yhat == cluster)
            cluster = list(zip(sources[row_ix, 0][0], sources[row_ix, 1][0]))
            # TODO: MAKE THIS NOT SO HACKY WITH THE NAME, MIGHT AVOID A PAIN LATER -> I THINK FIXED
            lines = graph_ax.plot(sources[row_ix, 1][0],
                                   sources[row_ix, 0][0],
                                   label=name if n == 0 else None)
            if not (min_size <= len(cluster) <= max_size):
                continue
            cluster_graphs.setdefault(name, []).append((cluster, lines))

    return cluster_graphs
