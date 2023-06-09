from matplotlib import pyplot as plt
from src.dataloader import DataLoader
from src.dimension_reduction import DimRed
from sklearn.metrics.pairwise import euclidean_distances, manhattan_distances

import numpy as np
import os
import pandas as pd
import scipy.cluster.hierarchy as sch


class Clustering:
    def __init__(self, dimred: DimRed, img_prefix: str, threshold: float, dist: str = "euclidean"):

        self.dimred = dimred

        #self.country_names = list(self.dimred.stand.dl.contact_data.keys())
        self.country_names = ['Ausztria', 'Belgium', 'Bulgária', 'Horvátország', 'Ciprus', 'Csehország', 'Dánia',
                              'Észtország', 'Finnország', 'Franciaország', 'Németország', 'Görögország', 'Magyarország',
                              'Írország', 'Olaszország', 'Lettország', 'Litvánia', 'Luxemburg', 'Málta', 'Hollandia',
                              'Lengyelország', 'Portugália', 'Románia', 'Szlovákia', 'Szlovénia', 'Spanyolország',
                              'Svédország', 'Albánia', 'Örményország', 'Fehéroroszország', 'Bosznia-Hercegovina',
                              'Izland', 'Észak-Macedónia', 'Szerbia', 'Svájc', 'Ukrajna', 'Egyesült Királyság',
                              'Montenegró', 'Oroszország']
        self.img_prefix = img_prefix
        self.threshold = threshold
        self.clusters = dict()
        self.final_clusters = dict()
        self.medoids = list()
        if dist == "euclidean":
            self.get_distance_matrix = self.get_euclidean_distance
        elif dist == "manhattan":
            self.get_distance_matrix = self.get_manhattan_distance

        os.makedirs("../plots", exist_ok=True)

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
        #Measure of closeness between countries before reordering
        plt.title("Az országok távolságának mértéke újrarendezés előtt",
                  fontsize=42, fontweight="bold")
        az = plt.imshow(distance, cmap='rainbow',
                        interpolation="nearest",
                        vmin=0)
        plt.colorbar(az)
        plt.savefig("../plots/" +
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

        self.get_clusters()

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
                self.clusters.update({x: list(dt.columns[order == x])})
                # print("cluster " + str(x) + ":", dt.columns[order == x])
        # Perform an indirect sort along the along first axis
        columns = [dt.columns.tolist()[i] for i in list((np.argsort(order)))]
        # Place columns(sorted countries) in the both axes
        dt = dt.reindex(columns, axis='index')
        dt = dt.reindex(columns, axis='columns')
        return columns, dt, res

    def plot_ordered_distance_matrix(self, columns, dt):
        plt.figure(figsize=(52, 45), dpi=300)
        az = plt.imshow(dt, cmap='rainbow',
                        alpha=.9, interpolation="nearest")
        plt.xticks(ticks=np.arange(len(columns)),
                   labels=columns,
                   rotation=90, fontsize=45)
        plt.yticks(ticks=np.arange(len(columns)),
                   labels=columns,
                   rotation=0, fontsize=45)
        cbar = plt.colorbar(az)
        tick_font_size = 115
        cbar.ax.tick_params(labelsize=tick_font_size)

        plt.savefig("../plots/" + self.img_prefix + "_" + "ordered_distance_1.pdf")

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
        #Cluster Analysis without threshold
        plt.title('Klaszteranalízis küszöbszám nélkül', fontsize=50, fontweight="bold")
        #Distance between Clusters
        plt.ylabel('Klaszterek távolsága', fontsize=45)
        plt.tight_layout()
        plt.savefig("../plots/" + self.img_prefix + "_" + "ordered_distance_2.pdf")

    def plot_dendrogram_with_threshold(self, res):
        fig, axes = plt.subplots(1, 1, figsize=(38, 24), dpi=300)
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
        #Cluster Analysis with a threshold
        #plt.title('Klaszteranalízis küszöbszámmal', fontsize=49, fontweight="bold")
        #Distance between Clusters
        plt.ylabel('.', fontsize=30)
        plt.tight_layout()
        axes.tick_params(axis='both', which='major', labelsize=26)
        plt.savefig("../plots/" + self.img_prefix + "_" + "ordered_distance_3.pdf")

    @staticmethod
    def _index(country: str, dl):
        i = 0
        #dl.age_data.keys()
        for cy in ["Belgium", "Bulgária", "Csehország", "Dánia", "Németország", "Észtország", "Írország", "Görögország",
                   "Spanyolország", "Franciaország", "Horvátország", "Olaszország", "Ciprus", "Lettország", "Litvánia",
                   "Luxemburg", "Magyarország", "Málta", "Hollandia", "Ausztria", "Lengyelország", "Portugália",
                   "Románia", "Szlovénia", "Szlovákia", "Finnország", "Svédország", "Albánia", "Örményország",
                   "Fehéroroszország", "Bosznia-Hercegovina", "Izland", "Észak-Macedónia", "Szerbia", "Svájc",
                   "Ukrajna", "Egyesült Királyság", "Montenegró", "Oroszország"]:
            if cy != country:
                i += 1
            else:
                break
        return i

    def get_clusters(self):
        columns, dt, res = self.calculate_ordered_distance_matrix()

        clus_and_feat = dict()
        dl = DataLoader()
        for i in self.clusters.keys():
            temp = dict()
            for country in self.clusters[i]:
                ind = self._index(country=country, dl=dl)
                features = self.dimred.data_cm_pca
                temp.update({country: features[ind]})
            clus_and_feat.update({i: temp})

        clusters = dict()
        for i in clus_and_feat.keys():
            clusters.update({i: list(clus_and_feat[i].keys())})

        medoids = list(0 for _ in range(len(clus_and_feat.keys())))
        for i in clus_and_feat.keys():
            sum = 0
            for country in clus_and_feat[i].keys():
                sum += clus_and_feat[i][country]
            center = sum / len(clus_and_feat[i])
            dists = np.zeros(len(clus_and_feat[i]))
            keys = list(clus_and_feat[i].keys())
            for k in range(len(clus_and_feat[i].keys())):
                dists[k] = np.linalg.norm(center - clus_and_feat[i][keys[k]])
            medoids[i - 1] = list(clus_and_feat[i].keys())[np.argmin(dists)]

        self.final_clusters = clusters
        self.medoids = medoids

        print("Clusters: ", self.final_clusters)
        print("Medoids of clusters: ", self.medoids)
        print(clus_and_feat)
