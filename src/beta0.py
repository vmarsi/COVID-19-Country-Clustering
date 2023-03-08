from src.dataloader import DataLoader
from src.model import RostModelHungary
from src.r0 import R0Generator

import numpy as np


class TransmissionRateCalc:
    def __init__(self, data: DataLoader, country: str, concept: str, base_r0: float = 2.2,
                 final_death_rate: float = 0.001):
        self.data = data
        self.country = country
        self.concept = concept
        self.base_r0 = base_r0
        self.final_death_rate = final_death_rate

        self.contact_mtx = self.get_contact_mtx()
        self.time_max = 0

    def run(self):
        self.data.model_parameters_data.update({"susc": np.ones(16) * 0.2})
        if self.concept == "base_r0":
            r0generator = R0Generator(param=self.data.model_parameters_data)
            self.time_max = 400
            return self.base_r0 / r0generator.get_eig_val(contact_mtx=self.contact_mtx)
        elif self.concept == "final_death_rate":
            self.time_max = 1000
            return self.death_beta_0()
        else:
            raise Exception("Provide a method for calculating beta_0.")

    def get_contact_mtx(self):
        contact_home = self.data.contact_data[self.country]["home"]
        contact_school = self.data.contact_data[self.country]["school"]
        contact_work = self.data.contact_data[self.country]["work"]
        contact_other = self.data.contact_data[self.country]["other"]
        contact_matrix = contact_home + contact_school + contact_work + contact_other
        return contact_matrix

    def death_beta_0(self):
        model = RostModelHungary(model_data=self.data, country=self.country, time_max=self.time_max)
        a = 0.01
        b = 1

        beta0 = 5
        while b - a > 0.01:
            self.data.model_parameters_data.update({"beta": a})
            sol = model.get_solution(t=model.time_vector, parameters=self.data.model_parameters_data,
                                     cm=self.contact_mtx)
            deaths_a = np.sum(model.get_deaths(solution=sol))

            self.data.model_parameters_data.update({"beta": b})
            sol = model.get_solution(t=model.time_vector, parameters=self.data.model_parameters_data,
                                     cm=self.contact_mtx)
            deaths_b = np.sum(model.get_deaths(solution=sol))

            c = (a + b) / 2
            self.data.model_parameters_data.update({"beta": c})
            sol = model.get_solution(t=model.time_vector, parameters=self.data.model_parameters_data,
                                     cm=self.contact_mtx)
            deaths_c = np.sum(model.get_deaths(solution=sol))

            if 0.001 * np.sum(model.population) - deaths_c == 0:
                beta0 = c
                break
            elif (0.001 * np.sum(model.population) - deaths_a) * (0.001 * np.sum(model.population) - deaths_c) < 0:
                b = c
            elif (0.001 * np.sum(model.population) - deaths_b) * (0.001 * np.sum(model.population) - deaths_c) < 0:
                a = c

        if not beta0 == 0:
            self.data.model_parameters_data.update({"beta": a})
            sol = model.get_solution(t=model.time_vector, parameters=self.data.model_parameters_data,
                                     cm=self.contact_mtx)
            deaths_a = np.sum(model.get_deaths(solution=sol))

            self.data.model_parameters_data.update({"beta": b})
            sol = model.get_solution(t=model.time_vector, parameters=self.data.model_parameters_data,
                                     cm=self.contact_mtx)
            deaths_b = np.sum(model.get_deaths(solution=sol))

            if abs(0.001 * np.sum(model.population) - deaths_a) < abs(0.001 * np.sum(model.population) - deaths_b):
                beta0 = a
            else:
                beta0 = b

        return beta0
