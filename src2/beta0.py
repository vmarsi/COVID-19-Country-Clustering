from src2.dataloader import DataLoader
from src2.model import RostModelHungary
from src2.r0 import R0Generator

import numpy as np


class TransmissionRateCalc:
    def __init__(self, data: DataLoader, country: str, concept: str, base_r0: float = 1):
        self.data = data
        self.country = country
        self.concept = concept
        self.base_r0 = base_r0

        self.contact_mtx = self.get_contact_mtx()

    def run(self):
        r0generator = R0Generator(param=self.data.model_parameters_data)
        if self.concept == "R0":
            return self.base_r0 / r0generator.get_eig_val(contact_mtx=self.contact_mtx)

    def get_contact_mtx(self):
        contact_home = self.data.contact_data[self.country]["home"]
        contact_school = self.data.contact_data[self.country]["school"]
        contact_work = self.data.contact_data[self.country]["work"]
        contact_other = self.data.contact_data[self.country]["other"]
        contact_matrix = contact_home + contact_school + contact_work + contact_other
        return contact_matrix
