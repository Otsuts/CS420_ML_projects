import argparse
import numpy as np
from datasets import get_data
from classifier import classifier
from utils import write_log, set_seed


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', default='mlp', type=str)
    parser.add_argument('--dataset', default='iris', type=str)
    parser.add_argument('--test_size', type=float, default=0.6)
    parser.add_argument('--C', type=int, default=2)
    parser.add_argument('--kernel', type=str, default='linear')
    parser.add_argument('--log',default='../logs')

    return parser.parse_args()


def main(args):
    x_train, x_test, y_train, y_test = get_data(args)
    clf = classifier(args=args)
    clf.fit(x_train, y_train)
    pred = clf.predict(x_test)

    write_log(
        f'Test precision {len(np.where(np.array(pred) == np.array(y_test))[0])/x_test.shape[0]:.4f}',args)


if __name__ == '__main__':
    set_seed(42)
    args = get_args()
    main(args)
