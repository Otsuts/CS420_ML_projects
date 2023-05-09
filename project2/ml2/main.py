import numpy as np
from sklearn.decomposition import FactorAnalysis
import math
import matplotlib.pyplot as plt
from config import *
from utils import save_data, load_data
from tqdm import tqdm


class FA_Test():
    def __init__(self, N=100, n=10, m=3, M=2, sigma=0.1, mu=None, A=None, verbose=True):
        if mu is None:
            self.mu = np.zeros([n, 1])
        else:
            self.mu = mu
        if A is None:
            self.A = np.random.rand(n, m)
        else:
            self.A = A
        self.N = N
        self.n = n
        self.real_m = m
        self.M = M
        self.sigma = sigma ** 0.5
        self.verbose = verbose
        self.generate_data()

    def generate_data(self):
        self.X = np.zeros([self.N, self.n])
        for i in range(self.N):
            # randomly generate the latent variable y
            self.y = np.random.randn(self.real_m, 1)

            # randomly generate the noise vector e
            self.e = np.random.randn(self.n, 1) * self.sigma

            # calculate the observed variable x
            x = (np.dot(self.A, self.y) + self.mu + self.e).squeeze()

            # add the sample to the data matrix X
            self.X[i, :] = x

    def main_loop(self):
        # try 2 * self.M + 1 different components values
        self.aic_score = np.zeros(2 * self.M + 1)
        self.bic_score = np.zeros(2 * self.M + 1)
        for i in range(self.real_m - self.M, self.real_m + self.M + 1):
            self.m = i
            fa_model = FactorAnalysis(n_components=self.m)
            fa_model.fit(self.X)
            self.log_likelihood = fa_model.score(self.X) * self.N
            self.aic_score[i - self.real_m + self.M] = self.AIC()
            self.bic_score[i - self.real_m + self.M] = self.BIC()
        if self.verbose:
            self.show_line()

    def free_para(self):
        # A mn, sigma 1 free parameters
        return self.m * self.n + 1

    def AIC(self):
        return self.log_likelihood - self.free_para()

    def BIC(self):
        return self.log_likelihood - 0.5 * self.free_para() * math.log(self.N, math.e)

    def show_line(self):
        plt.figure(figsize=(6, 6))
        plt.subplots_adjust(bottom=.10, top=0.95, hspace=.25, wspace=.15,
                            left=.3, right=.99)
        plt.subplot(2, 1, 1)
        plt.plot(range(self.real_m - self.M,
                 self.real_m + self.M + 1), self.aic_score)
        plt.xlabel("m components", fontsize=FONTSIZE)
        plt.ylabel("AIC", fontsize=FONTSIZE)
        plt.title("AIC/BIC-sklearn", fontsize=FONTSIZE)
        plt.subplot(2, 1, 2)
        plt.plot(range(self.real_m - self.M,
                 self.real_m + self.M + 1), self.bic_score)
        plt.xlabel("m components", fontsize=FONTSIZE)
        plt.ylabel("BIC", fontsize=FONTSIZE)
        plt.savefig(
            f'results/n{self.n}_m{self.real_m}_sigma{self.sigma:.4f}.png')


if __name__ == "__main__":
    params = []
    for n in [10,100,1000,10000]:
        for m in [3, 7, 10]:
            for sigma in [0.1, 1, 10]:
                params.append({'n': n, 'm': m, 'sigma': sigma})
    aic_accuracy = []
    bic_accuracy = []
    for param in params:
        test = FA_Test(verbose=True, **param)
        aic_choosen = []
        bic_choosen = []
        for _ in tqdm(range(MAX_ITE)):
            test.main_loop()  # directly use sklearn
            # test.EM_loop()  # private implementation
            aic_choosen.append(np.argmax(test.aic_score))
            bic_choosen.append(np.argmax(test.bic_score))
        aic_accuracy.append(
            {**param, 'acc': np.sum(np.array(aic_choosen) == 2) / MAX_ITE})  # self.M, in fact
        bic_accuracy.append(
            {**param, 'acc': np.sum(np.array(bic_choosen) == 2) / MAX_ITE})

    save_data(CRI_FILES[0], aic_accuracy)
    save_data(CRI_FILES[1], bic_accuracy)
