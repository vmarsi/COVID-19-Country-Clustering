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

    def run(self):
        self.dl.model_parameters_data.update({"calculated_beta_0": np.zeros(39)})
        i = 0
        for country in self.dl.contact_data.keys():
            beta0 = TransmissionRateCalc(data=self.dl, country=country, concept=self.concept, base_r0=self.base_r0,
                                         final_death_rate=self.final_death_rate)
            beta_calc = beta0.run()
            stand_mtx = beta_calc * beta0.contact_mtx
            self.dl.model_parameters_data["calculated_beta_0"][i] = beta_calc
            self.stand_mtxs.append(stand_mtx)
        self.stand_mtxs = np.array(self.stand_mtxs)


def main():
    dl = DataLoader()
    standardizer = Standardizer(dl=dl, concept="base_r0", base_r0=1.4)
    standardizer.run()
    print(standardizer.stand_mtxs)


if __name__ == "__main__":
    main()
