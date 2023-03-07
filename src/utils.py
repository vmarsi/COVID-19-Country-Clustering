import kmedoids
import matplotlib.pyplot as plt
import numpy as np
import os

import openpyxl
import xlrd

from src.beta0 import TransmissionRateCalc
from src.clustering import Clustering
from src.dataloader import DataLoader
from src.dimension_reduction import DimRed
from src.model import RostModelHungary
from src.standardizer import Standardizer

os.makedirs("../plots", exist_ok=True)


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
    hungary_contacts(stan=standardizer)
    country_contacts(stan=standardizer)


def main2():
    data = DataLoader()
    for country in ["Armenia", "Netherlands", "Germany"]:
        trc = TransmissionRateCalc(data=data, country=country, concept="base_r0")
        stan = Standardizer(dl=data, concept="base_r0")
        stan.run()
        model = RostModelHungary(model_data=data, country=country)
        stan.dl.model_parameters_data.update({"beta": stan.data_all_dict[country]["beta"]})
        sol = model.get_solution(t=model.time_vector, parameters=stan.dl.model_parameters_data, cm=trc.contact_mtx)
        incidence = np.diff(model.get_cumulative(solution=sol))
        plt.plot(model.time_vector[1:], incidence / data.age_data[country]["pop"])
    plt.show()


def main3():
    dl = DataLoader()
    standardizer = Standardizer(dl=dl, concept="base_r0")
    standardizer.run()
    dimred = DimRed(stand=standardizer, dim_red="2D2PCA")
    dimred.run()
    clustering = Clustering(dimred=dimred, img_prefix="2dpca_", threshold=1.4)
    columns, dt, res = clustering.calculate_ordered_distance_matrix()

    clus_and_feat = dict()
    for i in clustering.clusters.keys():
        temp = dict()
        for country in clustering.clusters[i]:
            ind = index(country=country)
            features = dimred.apply_dpca()
            temp.update({country: features[ind]})
        clus_and_feat.update({i: temp})
    print(clus_and_feat)

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
        min = np.min(dists)
        j = 0
        for _ in range(len(dists)):
            if dists[_] != min:
                j += 1
            else:
                break
        medoids[i-1] = list(clus_and_feat[i].keys())[j]
    print("Medoids: ", medoids)


def main4():
    dl = DataLoader()
    wb = xlrd.open_workbook("../data/gdp.xls")
    sheet = wb.sheet_by_index(0)
    wb.unload_sheet(0)
    wbd = dict()
    for country in dl.contact_data.keys():
        for i in range(sheet.nrows):
            if str(sheet.cell(i, 0)) == "text:'" + country + "'":
                wbd.update({country: [sheet.cell_value(i, sheet.ncols-1)]})
    wb = xlrd.open_workbook("../data/life_expectancy.xls")
    sheet = wb.sheet_by_index(0)
    wb.unload_sheet(0)
    for country in dl.contact_data.keys():
        for i in range(sheet.nrows):
            if str(sheet.cell(i, 0)) == "text:'" + country + "'":
                wbd[country].append(sheet.cell_value(i, sheet.ncols - 2))
    print(wbd)


def main5():
    data = DataLoader()
    data.model_parameters_data.update({"susc": np.ones(16) * 0.2})
    model = RostModelHungary(model_data=data, country="Hungary", time_max=1000)
    a = 0.01
    b = 1
    contact_home = data.contact_data["Hungary"]["home"]
    contact_school = data.contact_data["Hungary"]["school"]
    contact_work = data.contact_data["Hungary"]["work"]
    contact_other = data.contact_data["Hungary"]["other"]
    contact_matrix = contact_home + contact_school + contact_work + contact_other

    beta0 = 5
    while b - a > 0.01:
        data.model_parameters_data.update({"beta": a})
        sol = model.get_solution(t=model.time_vector, parameters=data.model_parameters_data,
                                 cm=contact_matrix)
        deaths_a = np.sum(model.get_deaths(solution=sol))

        data.model_parameters_data.update({"beta": b})
        sol = model.get_solution(t=model.time_vector, parameters=data.model_parameters_data,
                                 cm=contact_matrix)
        deaths_b = np.sum(model.get_deaths(solution=sol))

        c = (a+b)/2
        data.model_parameters_data.update({"beta": c})
        sol = model.get_solution(t=model.time_vector, parameters=data.model_parameters_data,
                                 cm=contact_matrix)
        deaths_c = np.sum(model.get_deaths(solution=sol))

        if 0.001 * np.sum(model.population) - deaths_c == 0:
            beta0 = c
            break
        elif (0.001 * np.sum(model.population) - deaths_a) * (0.001 * np.sum(model.population) - deaths_c) < 0:
            b = c
        elif (0.001 * np.sum(model.population) - deaths_b) * (0.001 * np.sum(model.population) - deaths_c) < 0:
            a = c

    if not beta0 == 0:
        data.model_parameters_data.update({"beta": a})
        sol = model.get_solution(t=model.time_vector, parameters=data.model_parameters_data,
                                 cm=contact_matrix)
        deaths_a = np.sum(model.get_deaths(solution=sol))

        data.model_parameters_data.update({"beta": b})
        sol = model.get_solution(t=model.time_vector, parameters=data.model_parameters_data,
                                 cm=contact_matrix)
        deaths_b = np.sum(model.get_deaths(solution=sol))

        if abs(0.001 * np.sum(model.population) - deaths_a) < abs(0.001 * np.sum(model.population) - deaths_b):
            beta0 = a
        else:
            beta0 = b
    return beta0


if __name__ == "__main__":
    main5()
