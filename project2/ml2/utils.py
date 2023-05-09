import matplotlib.pyplot as plt
import json
from config import *



def save_data(filename, data):
    if "txt" in filename:
        with open(filename, "w") as f:
            for i in range(data.shape[0]):
                f.write("{},{}\n".format(data[i, 0], data[i, 1]))
    elif "json" in filename:
        with open(filename, "w") as f:
            f.write(json.dumps(data))
    elif "npy" in filename:
        data = np.array(data)
        np.save(filename, data)


def load_data(filename):
    data = []
    if "txt" in filename:
        with open(filename, "r") as f:
            for line in f.readlines():
                data.append([float(item) for item in line.split(',')])
        return np.array(data)
    elif "json" in filename:
        with open(filename, "r") as f:
            data = json.loads(f.read())
        return data
    elif "npy" in filename:
        data = np.load(filename)
        return data
