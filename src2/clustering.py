import os

from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
import scipy.cluster.hierarchy as sch
from sklearn.metrics.pairwise import euclidean_distances, manhattan_distances

from src2.dataloader import DataLoader
from src2.dimension_reduction import DimRed
from src2.standardizer import Standardizer


class Clustering:
    def __init__(self, dimred: DimRed, img_prefix: str, threshold: float, dist: str = "euclidean"):

        self.dimred = dimred
        self.dimred.run()

        self.country_names = list(self.dimred.stand.dl.contact_data.keys())
        self.img_prefix = img_prefix
        self.threshold = threshold
        if dist == "euclidean":
            self.get_distance_matrix = self.get_euclidean_distance
        elif dist == "manhattan":
            self.get_distance_matrix = self.get_manhattan_distance

        os.makedirs("../plots2", exist_ok=True)

    def get_manhattan_distance(self):
        """
        Calculates Manhattan distance of a 39 * 136 matrix and returns 39*39 distance matrix
        :return matrix: square distance matrix with zero diagonals
        """
        manhattan_distance = manhattan_distances(self.dimred.data_cm_pca)  # get pairwise manhattan distance
        # convert the data into dataframe
        # replace the indexes of the distance with the country names
        # rename the columns and rows of the distance with country names and return a matrix distance
        dt = pd.DataFrame(manhattan_distance,
                          index=self.country_names, columns=self.country_names)
        return dt, manhattan_distance

    def get_euclidean_distance(self) -> np.array:
        """
        Calculates euclidean distance of a 39 * 136 matrix and returns 39*39 distance matrix
        :return matrix: square distance matrix with zero diagonals
        """
        # convert the data into dataframe
        euc_distance = euclidean_distances(self.dimred.data_cm_pca)
        dt = pd.DataFrame(euc_distance,
                          index=self.country_names, columns=self.country_names)  # rename rows and columns
        return dt, euc_distance

    def plot_distances(self):
        distance, _ = self.get_distance_matrix()
        plt.figure(figsize=(36, 28))
        plt.xticks(ticks=np.arange(len(self.country_names)),
                   labels=self.country_names,
                   rotation=90, fontsize=24)
        plt.yticks(ticks=np.arange(len(self.country_names)),
                   labels=self.country_names,
                   rotation=0, fontsize=24)
        plt.title("Measure of closeness  between countries before reordering",
                  fontsize=42, fontweight="bold")
        az = plt.imshow(distance, cmap='rainbow',
                        interpolation="nearest",
                        vmin=0)
        plt.colorbar(az)
        plt.savefig("../plots2/" +
                    self.img_prefix + "_" + "distances.pdf")

    def run(self):
        # calculate ordered distance matrix
        columns, dt, res = self.calculate_ordered_distance_matrix()

        # plot ordered distance matrix
        self.plot_ordered_distance_matrix(columns=columns, dt=dt)

        #  Original uncolored Dendrogram
        self.plot_dendrogram(res=res)

        self.plot_distances()

        #  Colored Dendrogram based on threshold (4 clusters)
        # cutting the dendrogram where the gap between two successive merges is at the largest.
        #  horizontal   line is drawn   through it.
        self.plot_dendrogram_with_threshold(res=res)

    def calculate_ordered_distance_matrix(self, verbose: bool = True):
        dt, distance = self.get_distance_matrix()
        # Return a copy of the distance collapsed into one dimension.
        distances = distance[np.triu_indices(np.shape(distance)[0], k=1)].flatten()
        #  Perform hierarchical clustering using complete method.
        res = sch.linkage(distances, method="complete")
        #  flattens the dendrogram, obtaining as a result an assignation of the original data points to single clusters.
        order = sch.fcluster(res, self.threshold, criterion='distance')
        if verbose:
            for x in np.unique(order):
                print("cluster " + str(x) + ":", dt.columns[order == x])
        # Perform an indirect sort along the along first axis
        columns = [dt.columns.tolist()[i] for i in list((np.argsort(order)))]
        # Place columns(sorted countries) in the both axes
        dt = dt.reindex(columns, axis='index')
        dt = dt.reindex(columns, axis='columns')
        return columns, dt, res

    def plot_ordered_distance_matrix(self, columns, dt):
        plt.figure(figsize=(45, 35), dpi=300)
        az = plt.imshow(dt, cmap='rainbow',
                        alpha=.9, interpolation="nearest")
        plt.xticks(ticks=np.arange(len(columns)),
                   labels=columns,
                   rotation=90, fontsize=32)
        plt.yticks(ticks=np.arange(len(columns)),
                   labels=columns,
                   rotation=0, fontsize=45)
        cbar = plt.colorbar(az)
        tick_font_size = 115
        cbar.ax.tick_params(labelsize=tick_font_size)

        plt.savefig("../plots2/" + self.img_prefix + "_" + "ordered_distance_1.pdf")

    def plot_dendrogram(self, res):
        fig, axes = plt.subplots(1, 1, figsize=(35, 25), dpi=150)
        sch.dendrogram(res,
                       leaf_rotation=90,
                       leaf_font_size=25,
                       labels=np.array(self.country_names),
                       orientation="top",
                       show_leaf_counts=True,
                       distance_sort=True)
        axes.tick_params(axis='both', which='major', labelsize=26)
        plt.title('Cluster Analysis without threshold', fontsize=50, fontweight="bold")
        plt.ylabel('Distance between Clusters', fontsize=45)
        plt.tight_layout()
        plt.savefig("../plots2/" + self.img_prefix + "_" + "ordered_distance_2.pdf")

    def plot_dendrogram_with_threshold(self, res):
        fig, axes = plt.subplots(1, 1, figsize=(35, 24), dpi=300)
        sch.dendrogram(res,
                       color_threshold=self.threshold,  # sets the color of the links above the color_threshold
                       leaf_rotation=90,
                       leaf_font_size=24,  # the size based on the number of nodes in the dendrogram.
                       show_leaf_counts=True,
                       labels=np.array(self.country_names),
                       above_threshold_color='blue',
                       ax=axes,
                       orientation="top",
                       get_leaves=True,
                       distance_sort=True)
        plt.title('Cluster Analysis with a threshold', fontsize=49, fontweight="bold")
        plt.ylabel('Distance between Clusters', fontsize=30)
        plt.tight_layout()
        axes.tick_params(axis='both', which='major', labelsize=26)
        plt.savefig("../plots2/" + self.img_prefix + "_" + "ordered_distance_3.pdf")


def main():
    dl = DataLoader()
    standardizer = Standardizer(dl=dl, concept="base_r0", base_r0=1.4)
    dimred = DimRed(stand=standardizer, dim_red="PCA")
    clustering = Clustering(dimred=dimred, img_prefix="pca_", threshold=0.25)
    clustering.run()


if __name__ == "__main__":
    main()
