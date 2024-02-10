# Run This File To Preform Anylisis On The Data

import json

with open("data/results/gage_test_save.json") as f:
    data = json.load(f)

import time
import warnings
from itertools import cycle, islice

import matplotlib.pyplot as plt
import numpy as np

from sklearn import cluster, mixture, metrics
from sklearn.neighbors import kneighbors_graph
from sklearn.preprocessing import StandardScaler

import read_csv

# ============
# Fetch Data Set
# ============
schools = read_csv.get_schools("data/SchoolList.csv")
students = read_csv.get_students("data/StudentsNEW.csv")


def get_student_by_school(school_name):
    X = []
    y = []
    for student in students:
        if student.school.name == school_name:
            X.append([student.house.lat, student.house.long])
    return np.array(X), y


gage = get_student_by_school("Gage")
washington = get_student_by_school("Washington")
century = get_student_by_school("Century")
dakota = get_student_by_school("Dakota Middle School")
riverside = get_student_by_school("Riverside")
lincoln = get_student_by_school("Lincoln K-8 Choice")

# ============
# Set up cluster parameters
# ============
plt.figure(figsize=(9 * 2 + 3, 13))
plt.subplots_adjust(
    left=0.02, right=0.98, bottom=0.001, top=0.95, wspace=0.05, hspace=0.01
)

plot_num = 1

default_base = {
    "quantile": 0.3,
    "eps": 0.18,
    "damping": 0.7,
    "preference": -200,
    "n_neighbors": 2,
    "min_samples": 7,
    "xi": 0.01,
    "min_cluster_size": None,
    "allow_single_cluster": False,
    "hdbscan_min_cluster_size": 2,
    "hdbscan_min_samples": 3,
    "random_state": 42,
}

test_sets = [gage, washington, century, dakota, riverside, lincoln]

model_performance = {}

for i_dataset, dataset in enumerate(test_sets):
    # update parameters with dataset-specific values
    params = default_base.copy()
    # params.update(algo_params)
    params["n_clusters"] = max(len(dataset[0]) // 40, 1)

    X, y = dataset

    # normalize dataset for easier parameter selection
    X = StandardScaler().fit_transform(X)

    # estimate bandwidth for mean shift
    bandwidth = cluster.estimate_bandwidth(X, quantile=params["quantile"])

    # connectivity matrix for structured Ward
    connectivity = kneighbors_graph(
        X, n_neighbors=params["n_neighbors"], include_self=False
    )
    # make connectivity symmetric
    connectivity = 0.5 * (connectivity + connectivity.T)

    # ============
    # Create cluster objects
    # ============
    # ms = cluster.MeanShift(bandwidth=bandwidth, bin_seeding=True)
    two_means = cluster.MiniBatchKMeans(
        n_clusters=params["n_clusters"],
        n_init="auto",  # Could Be Modified
        random_state=params["random_state"],  # Makes Deterministic
    )
    ward = cluster.AgglomerativeClustering(
        n_clusters=params["n_clusters"], linkage="ward", connectivity=connectivity
    )
    # spectral = cluster.SpectralClustering(
    #     n_clusters=params["n_clusters"],
    #     eigen_solver="arpack",  # No Reason To Change
    #     affinity="nearest_neighbors",  # Could Be Modified
    #     random_state=params["random_state"],  # Makes Deterministic
    # )
    # dbscan = cluster.DBSCAN(eps=params["eps"])
    # hdbscan = cluster.HDBSCAN(
    #     min_samples=params["hdbscan_min_samples"],
    #     min_cluster_size=params["hdbscan_min_cluster_size"],
    #     allow_single_cluster=params["allow_single_cluster"],
    # )
    # optics = cluster.OPTICS(
    #     min_samples=params["min_samples"],
    #     xi=params["xi"],
    #     min_cluster_size=params["min_cluster_size"],
    # )
    # affinity_propagation = cluster.AffinityPropagation(
    #     damping=params["damping"],
    #     preference=params["preference"],
    #     random_state=params["random_state"],
    # )
    average_linkage = cluster.AgglomerativeClustering(
        linkage="average",
        metric="cityblock",
        n_clusters=params["n_clusters"],
        connectivity=connectivity,
    )
    birch = cluster.Birch(n_clusters=params["n_clusters"])
    # gmm = mixture.GaussianMixture(
    #     n_components=params["n_clusters"],
    #     covariance_type="full",
    #     random_state=params["random_state"],
    # )

    clustering_algorithms = (
        ("MiniBatch\nKMeans", two_means),
        # ("Affinity\nPropagation", affinity_propagation),  # Not Consistent
        # ("MeanShift", ms),
        # ("Spectral\nClustering", spectral),
        ("Ward", ward),
        ("Agglomerative\nClustering", average_linkage),
        # ("DBSCAN", dbscan),  # Excludes Outliers
        # ("HDBSCAN", hdbscan),  # Excludes Outliers
        # ("OPTICS", optics),  # Excludes Outliers
        ("BIRCH", birch),
        # ("Gaussian\nMixture", gmm),
    )

    for name, algorithm in clustering_algorithms:
        t0 = time.time()

        # catch warnings related to kneighbors_graph
        with warnings.catch_warnings():
            warnings.filterwarnings(
                "ignore",
                message="the number of connected components of the "
                        + "connectivity matrix is [0-9]{1,2}"
                        + " > 1. Completing it to avoid stopping the tree early.",
                category=UserWarning,
            )
            warnings.filterwarnings(
                "ignore",
                message="Graph is not fully connected, spectral embedding"
                        + " may not work as expected.",
                category=UserWarning,
            )
            model = algorithm.fit(X)
            if not isinstance(algorithm, mixture.GaussianMixture):
                labels = model.labels_

                model_performance.setdefault(name, []).append((
                    metrics.silhouette_score(X, labels),
                    metrics.calinski_harabasz_score(X, labels),
                    metrics.davies_bouldin_score(X, labels)
                ))

                # print(f"Performance of {name}")
                # print(f"\tClosest To 1 Is Best: {metrics.silhouette_score(X, labels)}")
                # print(f"\tHighest Is Best: {metrics.calinski_harabasz_score(X, labels)}")
                # print(f"\tLowest Is Best: {metrics.davies_bouldin_score(X, labels)}")

        t1 = time.time()
        if hasattr(algorithm, "labels_"):
            y_pred = algorithm.labels_.astype(int)
        else:
            y_pred = algorithm.predict(X)

        plt.subplot(len(test_sets), len(clustering_algorithms), plot_num)
        if i_dataset == 0:
            plt.title(name, size=18)

        colors = np.array(
            list(
                islice(
                    cycle(
                        [
                            "#377eb8",
                            "#ff7f00",
                            "#4daf4a",
                            "#f781bf",
                            "#a65628",
                            "#984ea3",
                            "#999999",
                            "#e41a1c",
                            "#dede00",
                        ]
                    ),
                    int(max(y_pred) + 1),
                )
            )
        )
        # add black color for outliers (if any)
        colors = np.append(colors, ["#000000"])
        plt.scatter(X[:, 0], X[:, 1], s=10, color=colors[y_pred])

        plt.xlim(min(X[:, 0]), max(X[:, 0]))
        plt.ylim(min(X[:, 1]), max(X[:, 1]))
        plt.xticks(())
        plt.yticks(())
        plt.text(
            0.99,
            0.01,
            ("%.2fs" % (t1 - t0)).lstrip("0"),
            transform=plt.gca().transAxes,
            size=15,
            horizontalalignment="right",
        )
        plot_num += 1

for name, value in model_performance.items():
    print(f"Average Performance Of {name}")
    metrics = list(zip(*value))
    print(f"\tClosest To 1 Is Best: {sum(metrics[0]) / len(metrics[0])}")
    print(f"\tHighest Is Best: {sum(metrics[1]) / len(metrics[1])}")
    print(f"\tLowest Is Best: {sum(metrics[2]) / len(metrics[2])}")

plt.show()
