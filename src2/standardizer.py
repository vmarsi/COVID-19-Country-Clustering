from src2.beta0 import TransmissionRateCalc
from src2.dataloader import DataLoader

import numpy as np


class Standardizer:
    def __init__(self, concept: str, base_r0: float = 1):
        self.concept = concept
        self.base_r0 = base_r0
        self.stand_mtx = []

    def run(self):
        for

