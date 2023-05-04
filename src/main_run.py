from src.clustering import Clustering
from src.dataloader import DataLoader
from src.dimension_reduction import DimRed
from src.standardizer import Standardizer


def main():
    dl = DataLoader()
    standardizer = Standardizer(dl=dl, concept="final_death_rate")
    standardizer.run()
    dimred = DimRed(stand=standardizer, dim_red="2D2PCA")
    dimred.run()
    clustering = Clustering(dimred=dimred, img_prefix="2d2pca_fdr_man_", threshold=15, dist="manhattan")
    clustering.run()


if __name__ == "__main__":
    main()
