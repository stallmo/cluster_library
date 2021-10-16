from scipy.spatial.distance import cdist


def knowledge_gap(centers1, centers2, metric='seuclidean') -> float:
    """
    The knowledge gap is defined as the sum over the distances between centers in <centers1> and the closest center in <centers2>.
    Distance is defined by <metric>.

    :param centers1: np.array of shape (n_clusters, n_features).
    :param centers2: np.array of shape (n_clusters, n_features).
    :param metric: str. Optional. Defines which metric to use for distance calculation. Can be anything that is accepted by  scipy.spatial.distance.cdist().
    :return: float. The knowledge gap.
    """

    return cdist(centers1, centers2, metric=metric).min(axis=1).sum()
