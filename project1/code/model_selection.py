from sklearn.mixture import GaussianMixture
import numpy as np
import warnings
import argparse

warnings.filterwarnings('ignore')


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--sample_size_start', type=int, default=50)
    parser.add_argument('--sample_size_end', type=int, default=300)
    parser.add_argument('--sample_size_step', type=int, default=50)
    parser.add_argument('--dimension_start', type=int, default=2)
    parser.add_argument('--dimension_end', type=int, default=10)
    parser.add_argument('--dimension_step', type=int, default=3)
    parser.add_argument('--num_k_start', type=int, default=1)
    parser.add_argument('--num_k_end', type=int, default=20)
    parser.add_argument('--num_k_step', type=int, default=1)
    parser.add_argument('--num_repeats', type=int, default=5)
    parser.add_argument('--method', default='vbem')
    parser.add_argument('--seed', default=114514, type=int)
    parser.add_argument('--data_clusters', default=10, type=int)
    return parser.parse_args()


def EM(method, sample_sizes, dimensions, num_k, num_repeats, data_clusters):
    J_scores = []
    log_likelihood = []
    for sample_size in sample_sizes:
        for dimension in dimensions:
            for num_cluster in num_k:
                # prepare data
                X = []
                for _ in range(data_clusters):
                    X.append(np.random.multivariate_normal(np.zeros(dimension), np.eye(dimension), size=sample_size))
                X = np.vstack(X)
                # train models
                gmm = GaussianMixture(n_components=num_cluster, covariance_type='full', n_init=num_repeats)
                gmm.fit(X)
                score = gmm.score(X)
                log_likelihood.append(score)
                # print(f'Log likelihood is {score:.4f} on sample size {sample_size}, dimension {dimension}, cluster number {num_cluster} and num init {num_repeats}')
                if method == 'bic':
                    J_scores.append(gmm.bic(X))
                if method == 'aic':
                    J_scores.append(gmm.aic(X))
                if method == 'vbem':
                    J_scores.append(gmm.lower_bound_)
            max_index = J_scores.index(max(J_scores))
            print(
                f'Max log likelihood is {log_likelihood[max_index]:.4f} on sample size {sample_size}, dimension {dimension} with best k={num_k[max_index]}')
            J_scores.clear()
            log_likelihood.clear()


def main(args):
    np.random.seed(args.seed)
    #  set up parameters
    sample_sizes = range(args.sample_size_start, args.sample_size_end, args.sample_size_step)
    dimensions = range(args.dimension_start, args.dimension_end, args.dimension_step)
    num_k = range(args.num_k_start, args.num_k_end, args.num_k_step)
    num_repeats = args.num_repeats
    EM(args.method, sample_sizes, dimensions, num_k, num_repeats, args.data_clusters)


if __name__ == '__main__':
    args = parse_args()
    main(args)
