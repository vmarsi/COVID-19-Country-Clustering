from src.beta0 import TransmissionRateCalc
from src.dataloader import DataLoader
from src.r0 import R0Generator

import numpy as np


class Standardizer:
    def __init__(self, dl: DataLoader, concept: str, base_r0: float = 2.2, final_death_rate: float = 0.01):
        self.dl = dl
        self.concept = concept
        self.base_r0 = base_r0
        self.final_death_rate = final_death_rate
        self.stand_mtxs = []
        self.data_all_dict = dict()

    def run(self):
        r0generator = R0Generator(param=self.dl.model_parameters_data)
        stand_mtxs_temp = []
        for country in self.dl.contact_data.keys():
            beta0 = TransmissionRateCalc(data=self.dl, country=country, concept=self.concept, base_r0=self.base_r0,
                                         final_death_rate=self.final_death_rate)
            beta_calc = beta0.run()
            stand_mtx = beta_calc * beta0.contact_mtx
            stand_mtxs_temp.append(stand_mtx)
            self.data_all_dict.update(
                {country: {"beta": beta_calc,
                           "contact_full": beta0.contact_mtx,
                           "calc_r0": beta_calc * r0generator.get_eig_val(contact_mtx=beta0.contact_mtx),
                           "contact_home": self.dl.contact_data[country]["home"],
                           "contact_school": self.dl.contact_data[country]["school"],
                           "contact_work": self.dl.contact_data[country]["work"],
                           "contact_other": self.dl.contact_data[country]["other"]
                           }
                 })
        self.stand_mtxs = np.array(stand_mtxs_temp)
