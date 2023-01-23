from sklearn.decomposition import PCA
from sklearn import preprocessing
from src2.standardizer import Standardizer

import numpy as np


class DimRed:
    """
        This is (2D)^2 PCA class that applies both row-row and column-column directions to perform dimension reduction.
        input: 39 countries each 16 * 16 matrix concatenated row wise and column wise
        output: 39 countries each 2 * 2 matrix, and 39 * 4 (2 * 2 flatten matrix)
    """
    def __init__(self, stand: Standardizer, dim_red: str):
        self.stand = stand
        self.dim_red = dim_red
        self.data_cm_pca = []

    def run(self):
        if self.dim_red == "PCA":
            cm_for_1dpca = self.get_mtx_for_1dpca(self.stand.stand_mtxs)
            pca = PCA(n_components=4)
            pca.fit(cm_for_1dpca)
            data_pca = pca.transform(cm_for_1dpca)
            print("Explained variance ratios:",
                  pca.explained_variance_ratio_,
                  "->", sum(pca.explained_variance_ratio_))
        elif self.dim_red == "2D2PCA":
            data_pca = self.apply_dpca()
        else:
            raise Exception("Provide a type for dimensionality reduction.")
        self.data_cm_pca = data_pca

    @staticmethod
    def transposed(data):
        transposed = []
        for _ in range(len(data)):
            data_t = data[_].T
            transposed.append(data_t)
        return np.array(transposed)

    @staticmethod
    def get_mtx_for_1dpca(data):
        upper_tri_indexes = np.triu_indices(16)
        mtx = []
        for _ in range(len(data)):
            data_tri = data[_][upper_tri_indexes]
            mtx.append(data_tri)
        return np.array(mtx)

    @staticmethod
    def preprocess_data(data):
        # center the data
        centered_data = data - np.mean(data, axis=0)
        # normalize data
        data_scaled = preprocessing.scale(centered_data)
        return data_scaled

    def column_pca(self, col_dim: int = 2):
        contact_matrix = self.stand.stand_mtxs
        stacked_cm = np.vstack(contact_matrix)
        data_scaled = self.preprocess_data(data=stacked_cm)
        pca_1 = PCA(n_components=col_dim)
        pca_1.fit(data_scaled)

        print("Explained variance ratios:", pca_1.explained_variance_ratio_,
              "->", sum(pca_1.explained_variance_ratio_), "Eigenvectors:",
              pca_1.components_,  # (col_dim, 16)
              "Singular values:", pca_1.singular_values_)  # col_dim leading eigenvalues

        # Projection matrix for row direction matrix
        proj_matrix_1 = pca_1.components_.T  # 16 * col_dim projection matrix 1

        return proj_matrix_1

    def row_pca(self, row_dim: int = 2):
        contact_matrix_transposed = self.transposed(self.stand.stand_mtxs)
        stacked_cm_t = np.vstack(contact_matrix_transposed)
        data_scaled_2 = self.preprocess_data(data=stacked_cm_t)

        pca_2 = PCA(n_components=row_dim)
        pca_2.fit(data_scaled_2)

        print("Explained variance ratios 2:", pca_2.explained_variance_ratio_,
              "->", sum(pca_2.explained_variance_ratio_), "Eigenvectors 2:",
              pca_2.components_,  # (row_dim, 16)
              "Singular values 2:", pca_2.singular_values_)  # row_dim leading eigenvalues
        # print("PC 2", pc2)

        # Projection matrix for column direction matrix
        proj_matrix_2 = pca_2.components_.T  # 16 * row_dim projection matrix 2
        return proj_matrix_2

    def apply_dpca(self):
        # Now split concatenated original data into 39 sub-arrays of equal size i.e. 39 countries.
        contact_matrix = self.stand.stand_mtxs
        stacked_cm = np.vstack(contact_matrix)
        data_scaled = self.preprocess_data(data=stacked_cm)
        split = np.array_split(data_scaled, 39)
        data_split = np.array(split)
        # Get projection matrix for column direction
        proj_matrix_1 = self.column_pca()
        # Get projection matrix for row direction
        proj_matrix_2 = self.row_pca()

        # Now apply (2D)^2 PCA simultaneously using projection matrix 1 and 2
        matrix = proj_matrix_1.T @ data_split @ proj_matrix_2

        # Now reshape the matrix to get desired 39 * 4
        features = matrix.reshape((39, 4))
        return features
