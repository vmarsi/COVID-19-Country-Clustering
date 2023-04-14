from src.clustering import Clustering
from src.dataloader import DataLoader
from src.dimension_reduction import DimRed
from src.standardizer import Standardizer


def main():
    dl = DataLoader()
    standardizer = Standardizer(dl=dl, concept="base_r0")
    standardizer.run()
    dimred = DimRed(stand=standardizer, dim_red="2D2PCA")
    dimred.run()
    clustering = Clustering(dimred=dimred, img_prefix="2d2pca_", threshold=7)
    clustering.run()


if __name__ == "__main__":
    main()
