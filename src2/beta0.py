from src2.dataloader import DataLoader
from src2.model import RostModelHungary
from src2.r0 import R0Generator

import numpy as np


class TransmissionRateCalc:
    def __init__(self, data: DataLoader, country: str, concept: str, base_r0: float = 1.4,
                 final_death_rate: float = 0.001):
        self.data = data
        self.country = country
        self.concept = concept
        self.base_r0 = base_r0
        self.final_death_rate = final_death_rate

        self.contact_mtx = self.get_contact_mtx()

    def run(self):
        self.data.model_parameters_data.update({"susc": np.ones(16) * 0.2})
        if self.concept == "base_r0":
            r0generator = R0Generator(param=self.data.model_parameters_data)
            return self.base_r0 / r0generator.get_eig_val(contact_mtx=self.contact_mtx)
        elif self.concept == "final_death_rate":
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
        model = RostModelHungary(model_data=self.data, country=self.country)
        betas = np.zeros(100)
        deaths = np.zeros(100)
        dicti = dict()
        i = 0
        for beta in np.arange(0.01, 1.01, 0.01):
            betas[i] = beta
            self.data.model_parameters_data.update({"beta": beta})
            sol = model.get_solution(t=model.time_vector, parameters=self.data.model_parameters_data,
                                     cm=self.contact_mtx)
            deaths[i] = np.sum(model.get_deaths(solution=sol))
            dicti.update({deaths[i]: i})
            i += 1
        deaths_diff = self.final_death_rate * np.sum(model.population) - deaths
        deaths_min_diff = min(deaths_diff[np.where(deaths_diff >= 0)])
        beta_0 = 0
        for k in range(len(deaths_diff)):
            if deaths_diff[k] == deaths_min_diff:
                beta_0 = round(betas[k], 3)
        return beta_0