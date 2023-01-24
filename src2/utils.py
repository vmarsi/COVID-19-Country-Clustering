import matplotlib.pyplot as plt
import numpy as np
import os

from src2.dataloader import DataLoader
from src2.standardizer import Standardizer

os.makedirs("../plots2", exist_ok=True)


def hungary_contacts(stan):
    for typ in ["contact_home", "contact_school", "contact_work", "contact_other", "contact_full"]:
        # home contact
        img = plt.imshow(stan.data_all_dict['Hungary'][typ],
                         cmap='jet', vmin=0, vmax=4, alpha=.9, interpolation="nearest")
        ticks = np.arange(0, 16, 2)
        if typ == 'contact_full':
            cbar = plt.colorbar(img)
            tick_font_size = 40
            cbar.ax.tick_params(labelsize=tick_font_size)
        plt.xticks(ticks, fontsize=24)
        plt.yticks(ticks, fontsize=24)
        plt.savefig("../plots2/" + "hungary_" + typ.split("contact_")[1] + ".pdf")


def country_contacts(stan):
    for country in ["Armenia", "Belgium", "Estonia"]:
        # contact matrix Armenia
        matrix_to_plot = stan.data_all_dict[country]["contact_full"] * \
            stan.data_all_dict[country]["beta"]
        img = plt.imshow(matrix_to_plot,
                         cmap='jet', vmin=0, vmax=0.2,
                         alpha=.9, interpolation="nearest")
        ticks = np.arange(0, 16, 2)
        plt.xticks(ticks, fontsize=20)
        plt.yticks(ticks, fontsize=20)
        if country == "Estonia":
            cbar = plt.colorbar(img)
            tick_font_size = 25
            cbar.ax.tick_params(labelsize=tick_font_size)
        plt.savefig("../plots2/" + country + ".pdf")


def main():
    dl = DataLoader()
    standardizer = Standardizer(dl=dl, concept="base_r0", base_r0=1.4)
    standardizer.run()
    hungary_contacts(stan=standardizer)
    country_contacts(stan=standardizer)


if __name__ == "__main__":
    main()
