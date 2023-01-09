# -*- coding: utf-8 -*-

import time
import random
import numpy as np


def euclidean(point, data):
    """
    Euclidean distance between point & data.
    Point has dimensions (m,), data has dimensions (n,m), and output will be of size (n,).
    """
    return np.sqrt(np.sum((point - data)**2, axis=1))


class KMeans:
    def __init__(self):
        self.centroids = []

    def _evaluate(self, coords):
        labels = []
        coords_by_cluster = {}
        coord_idxs_by_cluster = {}
        for centroid_idx, centroid in enumerate(self.centroids):
            coords_by_cluster.setdefault(centroid_idx, [])
            coord_idxs_by_cluster.setdefault(centroid_idx, [])
        for coord_idx, coord in enumerate(coords):
            dists = euclidean(coord, self.centroids)
            centroid_idx = np.argmin(dists)
            labels.append(centroid_idx)
            coords_in_cluster = coords_by_cluster.setdefault(centroid_idx, [])
            coord_idxs_in_cluster = coord_idxs_by_cluster.setdefault(centroid_idx, [])
            coords_in_cluster.append(coord)
            coord_idxs_in_cluster.append(coord_idx)
        # self.plot_clustered_locations(list(dict(sorted(coords_by_cluster.items())).values()))
        return labels, self.centroids, list(dict(sorted(coords_by_cluster.items())).values()), list(dict(sorted(coord_idxs_by_cluster.items())).values())

    def fit_kmeans(self, coords, n_clusters, max_iter=5000):
        # Initialize the centroids, using the "k-means++" method, where a random datapoint is selected as the first,
        # then the rest are initialized w/ probabilities proportional to their distances to the first
        # Pick a random point from train data for first centroid
        self.centroids = [random.choice(coords)]
        for _ in range(n_clusters - 1):
            # Calculate distances from points to the centroids
            dists = np.sum([euclidean(centroid, coords) for centroid in self.centroids], axis=0)
            # Normalize the distances
            # print('sum: ', np.sum(dists))
            # print('dists: ', dists)
            if np.sum(dists) > 0:
                dists /= np.sum(dists)
                # Choose remaining points based on their distances
                new_centroid_idx, = np.random.choice(range(len(coords)), size=1, p=dists)
            else:
                new_centroid_idx, = np.random.choice(range(len(coords)), size=1)
            # new_centroid_idx = np.argmax(dists)
            self.centroids += [coords[new_centroid_idx]]
        # Iterate, adjusting centroids until converged or until passed max_iter
        iteration = 0
        prev_centroids = None
        while np.not_equal(self.centroids, prev_centroids).any() and iteration < max_iter:
            # Sort each data point, assigning to nearest centroid
            sorted_points = [[] for _ in range(n_clusters)]
            for x in coords:
                dists = euclidean(x, self.centroids)
                centroid_idx = np.argmin(dists)
                sorted_points[centroid_idx].append(x)
            # Push current centroids to previous, reassign centroids as mean of the points belonging to them
            prev_centroids = self.centroids
            self.centroids = [np.mean(cluster, axis=0) for cluster in sorted_points]
            for i, centroid in enumerate(self.centroids):
                if np.isnan(centroid).any():  # Catch any np.nans, resulting from a centroid having no points
                    self.centroids[i] = prev_centroids[i]
            iteration += 1
        return self._evaluate(coords)

    def fit_kmeans_constrained(self, coords, n_clusters, size_min, size_max, max_iter=5000):
        # TODO
        pass

    def kmeans_with_locations(self, locs, n_clusters, max_iter=5000):
        locations = np.array([[loc.lat, loc.lng] for loc in locs])
        labels, centroids, coords_by_cluster, coord_idxs_by_cluster = self.fit_kmeans(locations, n_clusters, max_iter)
        return labels, centroids, coords_by_cluster, coord_idxs_by_cluster

    @staticmethod
    def plot_clustered_locations(coords_list):
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots()
        # ax.plot(127.038030, 37.518452, marker='x', linestyle='', label=-1)
        for seq, coords in enumerate(coords_list):
            coords = np.array(coords)
            y, x = coords.transpose()
            ax.plot(x, y, marker='o', linestyle='', label=seq)
        ax.legend(fontsize=12, loc='upper left')  # legend position
        plt.title('Plot of clustering output, len: {}'.format(len(coords_list)), fontsize=20)
        plt.xlabel('Longitude', fontsize=14)
        plt.ylabel('Latitude', fontsize=14)
        plt.show()


def test_clustering():
    from sklearn.preprocessing import StandardScaler
    from sklearn.datasets import make_blobs
    # Create a dataset of 2D distributions
    centers = 5
    coords, true_labels = make_blobs(n_samples=100, centers=centers, random_state=42)
    coords = StandardScaler().fit_transform(coords)
    # Fit centroids to dataset
    kmeans = KMeans()
    start = time.time()
    labels, centroids, coords_by_cluster, coord_idxs_by_cluster = kmeans.fit_kmeans(coords, n_clusters=5, max_iter=1000)
    print('time spent: ', time.time() - start)
    # View results
    print(labels)
    print(centroids)
    print(coords_by_cluster)
    print(coord_idxs_by_cluster)
    kmeans.plot_clustered_locations(coords_by_cluster)


if __name__ == '__main__':
    test_clustering()
