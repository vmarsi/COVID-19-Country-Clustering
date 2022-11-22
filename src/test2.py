import matplotlib.pyplot as plt
import numpy as np
from src.dataloader import DataLoader
from src.model2 import RostModelHungary
from src.r0 import R0Generator
import xlrd


def main():
    age_data_file = "C:/Users/Viktor/PycharmProjects/covid19-country-clustering/data/age_data.xlsx"
    wb = xlrd.open_workbook(age_data_file)
    sheet = wb.sheet_by_index(0)
    datalist = np.array([sheet.row_values(i)[1:] for i in range(1, sheet.nrows)])
    wb.unload_sheet(0)
    age_data = datalist
    print(age_data)
    print(age_data.shape[1])
    dicti = {"key1": [1, 2, 3, 4], "key2": [1, 2, 3, 4], "key3": [1, 2, 3, 4], "key4": [1, 2, 3, 4]}
    array = list(dicti.values())
    arr = np.array(array).flatten()
    print(arr)
    print(dicti["key1"])
    l = [dicti[keys] for keys in dicti.keys()]
    a = np.array(l).flatten()
    print(a)


def main2():
    data = DataLoader()
    model = RostModelHungary(model_data=data)
    full_contact_mx = np.sum([data.contact_data["Hungary"][i] for i in data.contact_data["Hungary"].keys()], axis=0)
    data.model_parameters_data.update({"beta": 0.1})
    data.model_parameters_data.update({"susc": np.ones(16) * 0.5})
    sol = model.get_solution(t=model.time_vector, parameters=data.model_parameters_data, cm=full_contact_mx)
    deaths = model.get_deaths(solution=sol)
    plt.plot(model.time_vector, deaths)
    plt.show()


def main3():
    data = DataLoader()
    model = RostModelHungary(model_data=data)
    full_contact_mx = np.sum([data.contact_data["Hungary"][i] for i in data.contact_data["Hungary"].keys()], axis=0)
    data.model_parameters_data.update({"susc": np.ones(16) * 0.2})
    betas = np.zeros(200)
    deaths = np.zeros(200)
    dicti = dict()
    i = 0
    for beta in np.arange(0.01, 1.005, 0.005):
        betas[i] = beta
        data.model_parameters_data.update({"beta": beta})
        sol = model.get_solution(t=model.time_vector, parameters=data.model_parameters_data, cm=full_contact_mx)
        deaths[i] = np.sum(model.get_deaths(solution=sol))
        i += 1
        dicti.update({deaths[i]: i})
    deaths_min = min(abs(deaths - 0.001 * np.sum(model.population)))
    for k in range(len(deaths)):
        if deaths[k] == 0.001 * np.sum(model.population) - deaths_min \
                or deaths[k] == deaths_min + 0.001 * np.sum(model.population):
            data.model_parameters_data.update({"beta": round(betas[k], 3)})
    print(data.model_parameters_data["beta"])


def main4():
    data = DataLoader()
    model = RostModelHungary(model_data=data)
    full_contact_mx = np.sum([data.contact_data["Hungary"][i] for i in data.contact_data["Hungary"].keys()], axis=0)
    data.model_parameters_data.update({"beta": 0.125})
    data.model_parameters_data.update({"susc": np.ones(16) * 0.2})
    sol = model.get_solution(t=model.time_vector, parameters=data.model_parameters_data, cm=full_contact_mx)
    deaths = model.get_deaths(solution=sol)
    print(np.sum(deaths))
    print(0.001 * np.sum(model.population))


if __name__ == "__main__":
    main3()
