from src2.beta0 import TransmissionRateCalc
from src2.dataloader import DataLoader


class Standardizer:
    def __init__(self, dl: DataLoader, concept: str, base_r0: float = 1.4, final_death_rate: float = 0.001):
        self.dl = dl
        self.concept = concept
        self.base_r0 = base_r0
        self.final_death_rate = final_death_rate
        self.stand_mtx = []

    def run(self):
        beta0 = TransmissionRateCalc(data=self.dl, country="Hungary", concept=self.concept, base_r0=self.base_r0,
                                     final_death_rate=self.final_death_rate)
        self.stand_mtx = beta0.run() * beta0.contact_mtx
        print(beta0.run(), beta0.contact_mtx, self.stand_mtx)


def main():
    dl = DataLoader()
    standardizer = Standardizer(dl=dl, concept="final_death_rate", final_death_rate=0.001)
    standardizer.run()


if __name__ == "__main__":
    main()
