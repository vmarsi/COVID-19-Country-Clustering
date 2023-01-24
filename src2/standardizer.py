from src2.beta0 import TransmissionRateCalc
from src2.dataloader import DataLoader

import numpy as np


class Standardizer:
    def __init__(self, dl: DataLoader, concept: str, base_r0: float = 1.4, final_death_rate: float = 0.001):
        self.dl = dl
        self.concept = concept
        self.base_r0 = base_r0
        self.final_death_rate = final_death_rate
        self.stand_mtxs = []
        self.data_all_dict = dict()

    def run(self):
        self.dl.model_parameters_data.update({"calculated_beta_0": np.zeros(39)})
        i = 0
        stand_mtxs_temp = []
        for country in self.dl.contact_data.keys():
            beta0 = TransmissionRateCalc(data=self.dl, country=country, concept=self.concept, base_r0=self.base_r0,
                                         final_death_rate=self.final_death_rate)
            beta_calc = beta0.run()
            stand_mtx = beta_calc * beta0.contact_mtx
            self.dl.model_parameters_data["calculated_beta_0"][i] = beta_calc
            stand_mtxs_temp.append(stand_mtx)
            self.data_all_dict.update(
                {country: {"beta": beta_calc,
                           "contact_full": beta0.contact_mtx,
                           "contact_home": self.dl.contact_data[country]["home"],
                           "contact_school": self.dl.contact_data[country]["school"],
                           "contact_work": self.dl.contact_data[country]["work"],
                           "contact_other": self.dl.contact_data[country]["other"]
                           }
                 })
        self.stand_mtxs = np.array(stand_mtxs_temp)
