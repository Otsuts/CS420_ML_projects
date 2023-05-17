import pandas as pd
import numpy as np
from lingam import LiNGAM
from dealtwins import getdata
import matplotlib.pyplot as plt
plt.style.use("ggplot")


if __name__ == '__main__':
    x_path = '../data/twin_pairs_X_3years_samesex.csv'
    x_csv = pd.read_csv(x_path)
    for name in x_csv.columns[2:]:
        print(f'deal with {name}')
        try:
            x, y = getdata(name)
            X = pd.DataFrame(np.asarray([x, y]).T, columns=["x", "y"])
            lingam = LiNGAM()
            lingam.fit(X)
            plt.clf()
            plt.scatter(x, y, color="#0E7AC4")
            plt.title("Scatter Plot of $x\_name$")
            plt.savefig(f"../results/{name}.png")
        except:
            print(f'{name} cannot finish')
