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
    def __init__(self, dimred: DimRed, img_prefix: str, dist: str = "euclidean"):

        self.dimred = dimred
        self.dimred.run()

        self.country_names = self.dimred.stand.dl.contact_data.keys()
        self.img_prefix = img_prefix
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


def main():
    dl = DataLoader()
    standardizer = Standardizer(dl=dl, concept="base_r0", base_r0=1.4)
    dimred = DimRed(stand=standardizer, dim_red="PCA")
    clustering = Clustering(dimred=dimred, img_prefix="pca_")
    clustering.plot_distances()
    dimred2 = DimRed(stand=standardizer, dim_red="2D2PCA")
    clustering2 = Clustering(dimred=dimred2, img_prefix="dpca_")
    clustering2.plot_distances()
    dimred3 = DimRed(stand=standardizer, dim_red="2D2PCA")
    clustering3 = Clustering(dimred=dimred3, img_prefix="dpca_manh", dist="manhattan")
    clustering3.plot_distances()


if __name__ == "__main__":
    main()
