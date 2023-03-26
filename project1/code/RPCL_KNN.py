import copy

import numpy as np
import matplotlib.pyplot as plt


def k_means(X, k, method='RPCL'):
    '''
    :param X: dataset: np.array of shape [number,2]
    :param k: preset number of centroids
    :param method: whether to use knn or RPCL-knn
    :return: labels and centroids
    '''
    n = X.shape[0]
    centroids = X[np.random.choice(n, k, replace=False)]
    old_centroids = copy.deepcopy(centroids)
    converge = False
    iterations = 0

    while not converge:
        # Assign points to nearest centroid
        labels = np.argmin(np.sum((X[:, None, :] - centroids[None, :, :]) ** 2, axis=2), axis=1)
        if method == 'KNN':
            for i in range(k):
                indices = np.where(labels == i)[0]
                if len(indices) == 0:
                    continue
                centroids[i] = X[indices].mean(axis=0)
        elif method == 'RPCL':
            # Rival Penalized Competitive Learning
            rival_penalties = []
            for i in range(k):
                indices = np.where(labels == i)[0]
                if len(indices) == 0:
                    continue
                rival_penalty = np.sum((centroids[i] - centroids) ** 2, axis=1)
                rival_penalties.append(tuple([np.argmax((centroids[i] - centroids) ** 2, axis=0),
                                              np.max((centroids[i] - centroids) ** 2, axis=0)]))
                # rival_penalty = np.max((centroids[i]-centroids)**2,axis=1)
                diversity_term = 1 / (k - 1)
                update = X[indices].mean(axis=0)
                centroids[i] += 0.3 * (update - centroids[i])  # - 0.01 * np.sum(rival_penalty) * diversity_term
            for index, data in rival_penalties:
                diversity_term = 1 / (k - 1)
                centroids[index] -= 0.01 * data * diversity_term
        converge = (centroids - old_centroids <= 0.1).all()
        old_centroids = copy.deepcopy(centroids)
        iterations += 1
    print(f'{method} converges after {iterations} iterations')
    return centroids, labels


if __name__ == '__main__':
    np.random.seed(123)
    # generate dataset with 3 clusters
    X = np.concatenate([
        np.random.multivariate_normal([2, 2], [[0.2, 0], [0, 0.2]], size=50),
        np.random.multivariate_normal([-2, 2], [[0.2, 0], [0, 0.2]], size=50),
        np.random.multivariate_normal([0, -2], [[0.2, 0], [0, 0.2]], size=50)
    ])

    # Run k-means clustering with k=3
    centroids, labels = k_means(X, 3, method='KNN')

    # Plot the final cluster assignments
    plt.title('KNN K=3')
    plt.scatter(X[:, 0], X[:, 1], c=labels)
    plt.scatter(centroids[:, 0], centroids[:, 1], s=100, marker='x', c='r')
    plt.show()

    # Run k-means clustering with RPCL and k=4
    centroids, labels = k_means(X, 4, method='RPCL')

    # Plot the final cluster assignments
    plt.title('RPCL K=4')
    plt.scatter(X[:, 0], X[:, 1], c=labels)
    plt.scatter(centroids[:, 0], centroids[:, 1], s=100, marker='x', c='r')
    plt.show()

    # Run k-means clustering with KNN and k=4
    centroids, labels = k_means(X, 4, method='KNN')

    # Plot the final cluster assignments
    plt.title('KNN K=4')
    plt.scatter(X[:, 0], X[:, 1], c=labels)
    plt.scatter(centroids[:, 0], centroids[:, 1], s=100, marker='x', c='r')
    plt.show()
