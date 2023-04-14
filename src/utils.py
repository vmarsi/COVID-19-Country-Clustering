import matplotlib.pyplot as plt
import numpy as np
import os

from src.beta0 import TransmissionRateCalc
from src.dataloader import DataLoader
from src.model import RostModelHungary
from src.standardizer import Standardizer

os.makedirs("../plots", exist_ok=True)


def country_contacts(stan, country):
    for typ in ["contact_home", "contact_school", "contact_work", "contact_other", "contact_full"]:
        img = plt.imshow(stan.data_all_dict[country][typ],
                         cmap='jet', vmin=0, vmax=4, alpha=.9, interpolation="nearest")
        ticks = np.arange(0, 16, 2)
        if typ == 'contact_full':
            cbar = plt.colorbar(img)
            tick_font_size = 30
            cbar.ax.tick_params(labelsize=tick_font_size)
        plt.xticks(ticks, fontsize=16)
        plt.yticks(ticks, fontsize=16)
        plt.savefig("../plots/" + country + "__" + typ.split("contact_")[1] + ".pdf")
        plt.clf()


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
        plt.savefig("../plots/" + "hungary_" + typ.split("contact_")[1] + ".pdf")


def other_contacts(stan):
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
        plt.savefig("../plots/" + country + ".pdf")


def index(country: str):
    dl = DataLoader()
    i = 0
    for cy in dl.age_data.keys():
        if cy != country:
            i += 1
        else:
            break
    return i


def main():
    dl = DataLoader()
    standardizer = Standardizer(dl=dl, concept="base_r0", base_r0=1.4)
    standardizer.run()
    for country in ["Hungary", "Austria", "Russia"]:
        country_contacts(stan=standardizer, country=country)


def main2():
    data = DataLoader()
    for country in ["Armenia", "Netherlands", "Germany"]:
        trc = TransmissionRateCalc(data=data, country=country, concept="base_r0")
        stan = Standardizer(dl=data, concept="base_r0")
        stan.run()
        model = RostModelHungary(model_data=data, country=country, time_max=400)
        stan.dl.model_parameters_data.update({"beta": stan.data_all_dict[country]["beta"]})
        sol = model.get_solution(t=model.time_vector, parameters=stan.dl.model_parameters_data, cm=trc.contact_mtx)
        incidence = np.diff(model.get_cumulative(solution=sol))
        plt.plot(model.time_vector[1:], incidence / data.age_data[country]["pop"])
    plt.show()


if __name__ == "__main__":
    main2()
