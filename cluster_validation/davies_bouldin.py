from scipy.spatial.distance import cdist
import numpy as np


def calculate_central_fuzzy_db(data, centers, assignment_matrix, _num_clusters):
    n_data_points = data.shape[0]

    S = []
    # First, calculate the "cluster spreads" for each cluster
    for i in range(_num_clusters):
        center_i = centers[i].reshape((-1, data.shape[1]))
        dists_to_center_i = cdist(data, center_i, metric='minkowski', p=2) ** 2
        assignment_clust_i = assignment_matrix[:, i].reshape(dists_to_center_i.shape)

        S_i = dists_to_center_i.sum()
        S_i /= n_data_points  #
        S_i = np.sqrt(S_i)
        avg_membership = assignment_clust_i.sum() / n_data_points
        S_i *= avg_membership
        S.append(S_i)

    # Second, calculate how well the centers are separated
    M = cdist(centers, centers, metric='minkowski', p=2)

    # Third, calculate the cluster separation index for each pair of clusters
    R = np.zeros(shape=(_num_clusters, _num_clusters))
    for i in range(_num_clusters):
        for j in range(_num_clusters):
            if i == j:
                # R[i,j] = 0
                continue
            else:
                R[i, j] = (S[i] + S[j]) / M[i, j]

    # Calculate fuzzy Davies Bouldin
    # I.e., for each cluster, identify the "worst separated" cluster and average for all clusters
    fuzzy_db = R.max(axis=1).sum() / _num_clusters
    return fuzzy_db


# calculate deviation in each cluster (S_{i})
def calculate_federated_fuzzy_db(_local_learners, _num_clusters):
    S_global = []  # global cluster spreads

    # Calculate the global cluster spreads with local learners data
    for i in range(_num_clusters):
        points_assigned_total = 0
        membership_total_i = 0
        sum_dists_i = 0

        for local_learner in _local_learners:
            center_i = local_learner.centers[i].reshape((-1, local_learner.client_data.shape[1]))
            dists_to_center_i = cdist(local_learner.client_data, center_i,
                                      metric='minkowski', p=2) ** 2
            sum_dists_i += dists_to_center_i.sum()
            points_assigned_total += local_learner.client_data.shape[0]
            membership_total_i += local_learner.get_center_support()[i]

        S_i = np.sqrt(1.0 * sum_dists_i / points_assigned_total)
        avg_membership = 1.0 * membership_total_i / points_assigned_total
        S_i *= avg_membership

        S_global.append(S_i)

    # distances between cluster centers (centers are the same for all local learners)
    M = cdist(local_learner.centers, local_learner.centers)

    # construct matrix R -> How well (or bad) are clusters separated for each pair of clusters
    R = np.zeros(shape=(_num_clusters, _num_clusters))
    for i in range(_num_clusters):
        for j in range(_num_clusters):
            if i == j:
                # R[i,j] = 0
                continue
            else:
                R[i, j] = (S_global[i] + S_global[j]) / M[i, j]

    # Calculate federated fuzzy Davies Bouldin -> Average over "worst case" for each cluster
    fuzzy_db = R.max(axis=1).sum() / _num_clusters
    return fuzzy_db
