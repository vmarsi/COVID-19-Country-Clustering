import os

from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
import scipy.cluster.hierarchy as sch
from sklearn.metrics.pairwise import euclidean_distances, manhattan_distances

from src2.dimension_reduction import DimRed


class Clustering:
    def __init__(self, dimred: DimRed, img_prefix: str,
                 dist: str = "euclidean"):
        self.dimred = dimred
        self.country_names = self.dimred.stand.dl.contact_data.keys()
        self.img_prefix = img_prefix
        if dist == "euclidean":
            self.get_distance_matrix = self.get_euclidean_distance
        elif dist == "manhattan":
            self.get_distance_matrix = self.get_manhattan_distance

        os.makedirs("../plots", exist_ok=True)