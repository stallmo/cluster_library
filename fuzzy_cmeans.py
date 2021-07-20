import numpy as np
from scipy.spatial import distance

import copy


class FuzzyCMeans:

    def __init__(self, num_clusters, m=2, max_iter=1000, tol=0.0001):
        self.__num_clusters = num_clusters
        self.__max_iter = max_iter
        self.__tol = tol  # epsilon
        self.__m = m

        # going to be set when predict() is called
        self.__U = None  # assignment/ membership matrix
        self.__c = None  # cluster centers

    @property
    def num_clusters(self):
        return self.__num_clusters

    @property
    def max_iter(self):
        return self.__max_iter()

    @property
    def tol(self):
        return self.__tol

    @property
    def centers(self):
        return self.__c

    def __init_membership(self, X, initialization):

        n_rows = X.shape[0]
        n_cols = self.__num_clusters

        if initialization == 'random':
            self.__U = np.random.rand(n_rows, n_cols)
            self.__U = self.__U / self.__U.sum(axis=1)[:, None]  # make sure each row sums to 1

        pass

    def __update_cluster_centers(self, X):

        denominators = self.__U.sum(axis=0)[:, None]
        centers = self.__U.T.dot(X)
        self.__c = centers / denominators
        pass

    def __calculate_cluster_membership(self, X):

        power = 2.0 / (self.__m - 1.0)
        dist_to_centers = distance.cdist(X, self.__c)

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
        :param initialization: str. (optional). Specifies how the initial membership assignment is to be performed. Only 'random' is implented yet.
        :return:
        """
        if not initialization in ['random']:
            raise NotImplementedError

        if self.__U is None:
            self.__init_membership(X, initialization)

        for iter in range(self.__max_iter):

            U_prev = copy.deepcopy(self.__U)  # we'll need that to check termination criterion
            self.__update_cluster_centers(X)
            # update membership matrix
            self.__U = self.__calculate_cluster_membership(X)
            # check if we're done
            if np.linalg.norm(self.__U - U_prev) < self.__tol:
                break
        pass

    def predict(self, X):
        """
        Returns the matrix U (i.e. cluster membership) for points in X.

        :param X:
        :return:
        """

        if self.__U is None:
            print('Please call the fit() method first.')

        return self.__calculate_cluster_membership(X)
