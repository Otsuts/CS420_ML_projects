import numpy as np
import os
from torch.utils.data import Dataset
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split


def get_data(args):
    if args.dataset == 'iris':
        return loadIris(args)
    elif args.dataset == 'glass':
        return loadGlass(args)
    elif args.dataset == 'awa2':
        return loadAwA2(args)


def loadIris(args):
    data = load_iris()
    x = data['data']
    y = data['target']
    xTrain, xTest, yTrain, yTest = train_test_split(
        x, y, test_size=args.test_size, random_state=42)
    return xTrain, xTest, yTrain, yTest


def loadAwA2(args):
    x, y = ReadAwA2File()
    xTrain, xTest, yTrain, yTest = train_test_split(
        x, y, test_size=args.test_size, random_state=42)
    return xTrain, xTest, yTrain, yTest


def loadGlass(args):
    x, y = ReadGlassFile()
    xTrain, xTest, yTrain, yTest = train_test_split(
        x, y, test_size=args.test_size, random_state=42)
    return xTrain, xTest, yTrain, yTest


def ReadGlassFile():
    with open("../datasets/glass.txt") as f:
        row = f.readline()
        row = row.split(' ')
        x = np.asarray([float(row[i].split(':')[1]) for i in range(1, 10)])
        y = np.asarray([int(row[0])-1])
        while row:
            row = f.readline()
            if row != '':
                row = row.split(' ')
                # print(row)
                x = np.vstack((x, [float(row[i].split(':')[1])
                                   for i in range(1, 10)]))
                y = np.vstack((y, [int(row[0])-1]))
    return x, y.ravel()


def ReadAwA2File():
    data_path = os.path.join(
        '../datasets/Animals_with_Attributes2/Features/ResNet101', 'AwA2-features.txt')
    label_path = os.path.join(
        '../datasets/Animals_with_Attributes2/Features/ResNet101', 'AwA2-labels.txt')
    data = []
    with open(data_path, 'r') as f:
        for line in f:
            data.append(list(map(float, line.strip().split())))

    data = np.asarray(data)
    labels = []
    with open(label_path, 'r') as f:
        for line in f:
            labels.append(int(line)-1)
    labels = np.asarray(labels)
    return data, labels


if __name__ == '__main__':
    x, y = ReadAwA2File()
    print(x.shape, y.shape)
