import matplotlib.pyplot as plt
import numpy as np

from src.dataloader import DataLoader
from src.model import RostHungaryModel
from src.r0 import R0Generator


def main():
    print(np.linspace(0, 100, 100))
    print(np.arange(10))

    k = np.zeros((10, 10))
    i = 1
    for n in range(len(k)):
        for m in range(len(k[0])):
            k[n][m] = i
            i += 1
    print(k)
    print(k[1:7:2, 2:9:3])
    k[1:7:2, 2:9:3] = [[1, 1, 1], [1, 1, 1], [1, 1, 1]]
    print(k)
    print(k[1:])

    a = [1, 2, 3]
    b = [8, 9]
    for (x, y) in zip(a, b):
        print(x, y)

    print([[1, 2], [3, 4]])

    u = np.zeros((10, 9))
    i = 1
    for n in range(len(u)):
        for m in range(len(u[0])):
            u[n][m] = i
            i += 1
    p, q, r = u.reshape(-1, 9)
    print(p)


def main2():
    data = DataLoader()
    data.model_parameters_data.update({"susc": np.array([1.0] * 16)})
    r0generator = R0Generator(param=data.model_parameters_data)
    x = r0generator.get_eig_val(contact_mtx=data.contact_data["Hungary"]["home"])
    print(x)

    data.model_parameters_data.update({"beta": 0.027})
    ic = {
        "s": data.age_data["Hungary"]["age"],
        "l1": [0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        "l2": [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        "ip": [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        "ia1": [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        "ia2": [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        "ia3": [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        "is1": [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        "is2": [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        "is3": [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        "ih": [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        "ic": [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        "icr": [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        "d": [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        "r": [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        "c": [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
        }
    ic_sum = np.sum(np.array([ic[k] for k in ic.keys() if k != "s"]), axis=0)
    ic["s"] = ic["s"] - ic_sum
    full_contact_mx = np.sum([data.contact_data["Hungary"][i] for i in data.contact_data["Hungary"].keys()], axis=0)
    # full_corrected_contact_mx = (((full_contact_mx.T * data.age_data["Hungary"]["age"]).T +
    #                             (full_contact_mx.T * data.age_data["Hungary"]["age"]).T).T / (2*data.age_data["Hungary"]["age"])).T
    model = RostHungaryModel(init_values=ic, contact_matrix=full_contact_mx,
                             parameters=data.model_parameters_data, to_solve=True)
    s_sol = model.solution[::, 0:15]
    s_sol_sum = np.sum(s_sol, axis=1)
    plt.plot(model.time_vector, s_sol_sum)
    plt.show()


if __name__ == "__main__":
    main()