from scipy.spatial import distance


def within_cluster_sse(X, assignment, centers):
    """
    Calculates the sum of squared errors between all points and its assigned cluster centers as a measure for cluster cohesion.

    :param X: np.array with the data points.
    :param assignment: 1-d iterable of length <number_of_examples>. Entry i is datapoint i's cluster label in 1, ..., K.
    :param centers: np.array with cluster centers. Entry j is center of cluster j.
    :return: float.
    """

    return sum([distance.cdist(X[assignment == i], center.reshape((1, -1))).sum() for i, center in enumerate(centers)])

def outside_cluster_sse(X, assignment, centers):
    """
    Calculates the sum of squared errors between all points and the centers of the clusters the point is not assigned to.
    A measure for cluster separation.

    :param X: np.array with the data points.
    :param assignment: 1-d iterable of length <number_of_examples>. Entry i is datapoint i's cluster label in 1, ..., K.
    :param centers: np.array with cluster centers. Entry j is center of cluster j.
    :return: float.
    """

    return sum([distance.cdist(X[assignment != i], center.reshape((1, -1))).sum() for i, center in enumerate(centers)])