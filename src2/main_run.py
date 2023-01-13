from src2.clustering import Clustering
from src2.dataloader import DataLoader
from src2.dimension_reduction import DimRed
from src2.standardizer import Standardizer


def main():
    dl = DataLoader()
    standardizer = Standardizer(dl=dl, concept="base_r0", base_r0=1.4)
    standardizer.run()
    dimred = DimRed(stand=standardizer, dim_red="PCA")
    dimred.run()
    clustering = Clustering(dimred=dimred, img_prefix="pca_", threshold=0.25)
    clustering.run()


if __name__ == "__main__":
    main()
