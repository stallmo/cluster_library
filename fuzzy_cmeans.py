import numpy as np
from scipy.spatial import distance

import copy


class FuzzyCMeans:

    def __init__(self, num_clusters, m=2, max_iter=1000, tol=0.0001):
        self.__num_clusters = num_clusters
        self.__max_iter = max_iter
        self.__tol = tol  # epsilon
        self.__m = m

        # going to be set when fit() is called
        self.__U = None  # assignment/ membership matrix
        self.__c = None  # cluster centers
        self.__iterations = 0

    @property
    def num_clusters(self):
        return self.__num_clusters

    @property
    def max_iter(self):
        return self.__max_iter

    @property
    def tol(self):
        return self.__tol

    @property
    def centers(self):
        return self.__c

    @property
    def iterations(self):
        return self.__iterations

    def set_centers(self, centers):
        self.__c = centers

    def __init_centers(self, X, initialization):

        data_dim = X.shape[1]

        if initialization == 'random_pick':
            # Randomly choose points from X and choose as centers
            indices = np.random.choice(X.shape[0], self.__num_clusters, replace=False)
            self.__c = X[indices, :]

        if initialization == 'random':
            data_min = X.min()
            data_max = X.max()
            self.__c = np.random.uniform(low=data_min, high=data_max, size=((self.__num_clusters, data_dim)))

        # implement kmeans ++ initialization
        if initialization == 'k++':
            # Choose first center randomly
            self.__c = [X[np.random.choice(X.shape[0])]]

            for _ in range(1, self.__num_clusters):
                # Calculate minimal distance of every point to already chosen centers
                dist_to_centers = distance.cdist(X, np.array(self.__c)).min(axis=1)

                # Probability distribution based on distances
                probabilities = dist_to_centers / dist_to_centers.sum()

                # Choose next center based on probability distribution
                next_center = X[np.random.choice(X.shape[0], p=probabilities)]
                self.__c.append(next_center)

            self.__c = np.array(self.__c)

        pass

    def __init_membership(self, X, initialization):

        n_rows = X.shape[0]
        n_cols = self.__num_clusters

        if initialization == 'random':
            self.__U = np.random.rand(n_rows, n_cols)
            self.__U = self.__U / self.__U.sum(axis=1)[:, None]  # make sure each row sums to 1
        pass

    def __update_cluster_centers(self, X):

        U_m = np.power(self.__U, self.__m)
        denominators = U_m.sum(axis=0)[:, None]
        centers = U_m.T.dot(X)
        self.__c = centers / denominators
        pass

    def __calculate_cluster_membership(self, X):

        power = 2.0 / (self.__m - 1.0)
        dist_to_centers = distance.cdist(X, self.__c)

        # Avoid division by zero
        epsilon = 1e-10
        dist_to_centers = np.maximum(dist_to_centers, epsilon)

        U_denom = np.zeros((X.shape[0], self.__num_clusters))

        # TODO: Optimize this bit
        for j in range(self.__num_clusters):
            cj_dists = np.repeat(dist_to_centers[:, j], self.__num_clusters).reshape(X.shape[0], -1)
            dists_normalized = np.power(cj_dists / dist_to_centers, power)
            U_denom[:, j] = dists_normalized.sum(axis=1)

        membership_matrix = 1.0 / U_denom

        return membership_matrix

    def fit(self, X, initialization='random'):
        """
        Fits fuzzy c-means to data X.

        :param X: 2d numpy array. Contains the training data. Each row is an observation.
        :param initialization: str. (optional).
         Specifies how the initial membership assignment is to be performed. Only 'random' is implemented
        :return: None.
        """

        if self.__iterations == 0:

            if not initialization in ['random', 'random_pick', 'k++', 'federated']:
                raise NotImplementedError

            if initialization == 'federated':
                assert self.__c is not None, 'For the federated initialization, the global server must first communicate the global cluster centers.'
                print('Using federated initialization. Using global cluster centers for initialization.')

            if self.__U is None:
                self.__init_centers(X, initialization)
                self.__U = self.__calculate_cluster_membership(X)
                # self.__init_membership(X, initialization)

        for _iter in range(self.__max_iter):

            self.__iterations += 1
            U_prev = copy.deepcopy(self.__U)  # we'll need that to check termination criterion
            # update membership matrix
            self.__U = self.__calculate_cluster_membership(X)
            self.__update_cluster_centers(X)
            # check if we're done
            diff = np.linalg.norm(self.__U - U_prev)
            if diff < self.__tol and self.__iterations > 1:
                print(
                    f'Converged after {self.__iterations} iterations, because the change in membership matrix was {diff:.6f} < {self.__tol}.')
                break
        pass

    def predict(self, X):
        """
        Returns matrix U (i.e. cluster membership) for points in X.

        :param X: numpy matrix of shape (num_examples, num_features)
        :return: numpy matrix with fuzzy cluster assignments of shape (num_examples, num_clusters).
        """

        if self.__U is None:
            raise RuntimeError('The model has not been fitted yet.')

        return self.__calculate_cluster_membership(X)
