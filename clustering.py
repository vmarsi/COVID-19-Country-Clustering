import numpy as np
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from scipy.spatial.distance import cdist
import matplotlib.pyplot as plt
from scipy.spatial.distance import pdist, squareform
from fastcluster import linkage

from dataloader import DataLoader
from plotter import Plotter
from simulation import Simulation


class DataTransformer:
    def __init__(self):
        self.data = DataLoader()

        self.susc = 1.0
        self.base_r0 = 2.2

        self.upper_tri_indexes = np.triu_indices(16)
        self.country_names = list(self.data.age_data.keys())

        self.data_all_dict = dict()
        self.data_mtx_dict = dict()
        self.data_clustering = []
        self.get_data_for_clustering()
        self.contact_matrix = None

    def get_data_for_clustering(self):
        global simulation
        for country in self.country_names:
            age_vector = self.data.age_data[country]["age"].reshape((-1, 1))
            contact_matrix = self.data.contact_data[country]["home"] + \
                self.data.contact_data[country]["work"] + \
                self.data.contact_data[country]["school"] + \
                self.data.contact_data[country]["other"]
            contact_home = self.data.contact_data[country]["home"]

            susceptibility = np.array([1.0] * 16)
            susceptibility[:4] = self.susc
            simulation = Simulation(data=self.data, base_r0=self.base_r0,
                                    contact_matrix=contact_matrix,
                                    contact_home=contact_home,
                                    age_vector=age_vector,
                                    susceptibility=susceptibility)
            self.data_all_dict.update(
                {country: {"beta": simulation.beta,
                           "age_vector": age_vector,
                           "contact_full": contact_matrix,
                           "contact_home": contact_home
                           }
                 })
            self.data_mtx_dict.update(
                {country: {"full": simulation.beta * contact_matrix[self.upper_tri_indexes],
                           "home": simulation.beta * contact_home[self.upper_tri_indexes]
                           }
                 })
            self.data_clustering.append(
                simulation.beta * contact_matrix[self.upper_tri_indexes])
        self.data_clustering = np.array(self.data_clustering)


class Clustering:
    def __init__(self, data):
        self.data = data
        self.n_cl = 3

        self.k_means_pred = None
        self.centroids = None
        self.closest_points = None
        self.closest_point_idx = None

    def run_clustering(self):
        k_means = KMeans(n_clusters=self.n_cl, random_state=1)
        k_means.fit(self.data)
        self.k_means_pred = k_means.predict(self.data)
        self.centroids = k_means.cluster_centers_

    def get_closest_points(self):
        self.closest_point_idx = (-1) * np.ones(self.n_cl).astype(int)
        for c_idx, centroid in enumerate(self.centroids):
            min_dist = None
            for idx, point in enumerate(self.data):
                if self.k_means_pred[idx] == c_idx:
                    dist = np.sum((point - centroid) ** 2)
                    if min_dist is None or dist < min_dist:
                        min_dist = dist
                        self.closest_point_idx[c_idx] = idx
                        self.closest_points = self.data[np.array(self.closest_point_idx).astype(int), :2]

    def get_distance_points(self, no_of_iterations):
        idx = np.random.choice(len(self.data), self.n_cl, replace=False)  # Randomly choosing Centroids
        centroids = self.data[idx, :]
        distances = cdist(self.data, centroids, 'euclidean')
        points = np.array([np.argmin(i) for i in distances])
        for _ in range(no_of_iterations):
            centroids = []
            for idx in range(self.n_cl):
                temp_cent = self.data[points == idx].mean(axis=0)
                centroids.append(temp_cent)
                centroids = np.vstack(centroids)
                distances = cdist(self.data, centroids, 'euclidean')
                points = np.array([np.argmin(i) for i in distances])
                return points

    def distance(self):
        dist_matrix = np.zeros(16, 16)
        length = self.data.shape
        for i in range(length[0]):
            for j in range(length[1]):
                dist_matrix = squareform(pdist(self.data))
        plt.imshow(dist_matrix)


def seriation(Z, N, cur_index):
    '''
        input:
            - Z is a hierarchical tree (dendrogram)
            - N is the number of points given to the clustering process
            - cur_index is the position in the tree for the recursive traversal
        output:
            - order implied by the hierarchical tree Z

        seriation computes the order implied by a hierarchical tree (dendrogram)
    '''
    if cur_index < N:
        return [cur_index]
    else:
        left = int(Z[cur_index - N, 0])
        right = int(Z[cur_index - N, 1])
        return seriation(Z, N, left) + seriation(Z, N, right)


def compute_serial_matrix(dist_matrix, method="ward"):
    '''
        input:
            - dist_mat is a distance matrix
            - method = ["ward","single","average","complete"]
        output:
            - seriated_dist is the input dist_mat,
              but with re-ordered rows and columns
              according to the seriation, i.e. the
              order implied by the hierarchical tree
            - res_order is the order implied by
              the hierarhical tree
            - res_linkage is the hierarhical tree (dendrogram)

        compute_serial_matrix transforms a distance matrix into
        a sorted distance matrix according to the order implied
        by the hierarchical tree (dendrogram)
    '''
    N = len(dist_matrix)
    flat_dist_matrix = squareform(dist_matrix)
    res_linkage = linkage(flat_dist_matrix, method=method, preserve_input=True)
    res_order = seriation(res_linkage, N, N + N - 2)
    seriated_dist = np.zeros((N, N))
    a, b = np.triu_indices(N, k=1)
    seriated_dist[a, b] = dist_matrix[[res_order[i] for i in a], [res_order[j] for j in b]]
    seriated_dist[b, a] = seriated_dist[a, b]
    return seriated_dist, res_order, res_linkage


def main():
    # Create data for clustering
    data_tr = DataTransformer()

    # Reduce dimensionality
    pca = PCA(n_components=12)
    pca.fit(data_tr.data_clustering)
    data_pca = pca.transform(data_tr.data_clustering)
    print("Explained variance ratios:", pca.explained_variance_ratio_,
          "->", sum(pca.explained_variance_ratio_))
    print("data_clustering:", data_tr.data_clustering)
    length = len(data_tr.data_clustering)
    dist_matrix = squareform(pdist(data_tr.data_clustering))
    plt.pcolormesh(dist_matrix)
    plt.colorbar()
    plt.xlim([0, length])
    plt.ylim([0, length])
    plt.show()
    print("The distance matrix:", dist_matrix)
    label = KMeans(data_pca)
    u_labels = np.unique(label)
    for i in u_labels:
        plt.scatter(data_pca[label == i, 0], data_pca[label == i, 1], label=i)
        plt.legend()
        plt.show()
    print(len(dist_matrix))

    methods = ["ward", "single", "average", "complete"]
    for method in methods:
        print("Method:\t", method)
        ordered_dist_matrix, res_order, res_linkage = compute_serial_matrix(dist_matrix, method)
        plt.pcolormesh(ordered_dist_matrix)
        plt.colorbar()
        plt.xlim([0, len(dist_matrix)])
        plt.ylim([0, len(dist_matrix)])
        plt.show()
        print(squareform(dist_matrix))

    # Execute clustering
    clust = Clustering(data=data_pca)
    clust.run_clustering()
    clust.get_closest_points()

    # Plot results for analysis
    plotter = Plotter(clustering=clust,
                      data_transformer=data_tr)
    plotter.plot_clustering()
    centroids_orig = pca.inverse_transform(clust.centroids)
    plotter.plot_heatmap_centroid(centroids=centroids_orig)
    plotter.plot_heatmap_closest()

    # List cluster members
    for cluster in range(clust.n_cl):
        print("Cluster", cluster, "(" + plotter.colors[cluster] + ")", ":",
              {data_tr.country_names[idx]: data_tr.data_all_dict[data_tr.country_names[idx]]["beta"]
               for idx, x in enumerate(clust.k_means_pred) if x == cluster})


if __name__ == "__main__":
    main()
