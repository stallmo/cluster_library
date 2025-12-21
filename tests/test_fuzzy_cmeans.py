# python
import numpy as np
import pytest
from cluster_library import FuzzyCMeans


def test_init_properties():
    m = 2.5
    model = FuzzyCMeans(num_clusters=3, m=m, max_iter=10, tol=1e-5)
    assert model.num_clusters == 3
    assert model.max_iter == 10
    assert model.tol == 1e-5
    assert model.centers is None
    assert model.iterations == 0


def test_predict_before_fit_raises():
    X = np.array([[0.0, 0.0]])
    model = FuzzyCMeans(num_clusters=2)
    with pytest.raises(RuntimeError):
        model.predict(X)


def test_invalid_initialization_raises():
    X = np.vstack([np.zeros((5, 2)), np.ones((5, 2))])
    model = FuzzyCMeans(num_clusters=2)
    with pytest.raises(NotImplementedError):
        model.fit(X, initialization='unsupported_init')


def test_random_initialization_converges_and_outputs_shapes():
    np.random.seed(0)
    cluster1 = np.random.normal(loc=0.0, scale=0.1, size=(10, 2))
    cluster2 = np.random.normal(loc=1.0, scale=0.1, size=(10, 2))
    X = np.vstack([cluster1, cluster2])

    model = FuzzyCMeans(num_clusters=2, max_iter=100, tol=1e-4)
    np.random.seed(43)
    model.fit(X, initialization='random')

    U = model.predict(X)
    assert U.shape == (X.shape[0], 2)
    assert np.allclose(U.sum(axis=1), 1.0, atol=1e-6)
    assert model.centers is not None
    assert model.centers.shape == (2, X.shape[1])
    assert 1 <= model.iterations <= model.max_iter


def test_random_pick_initialization_fit():
    np.random.seed(7)
    X = np.vstack([np.random.normal(loc=i, scale=0.01, size=(5, 2)) for i in range(3)])
    model = FuzzyCMeans(num_clusters=3, max_iter=100)
    np.random.seed(123)
    model.fit(X, initialization='random_pick')
    centers = model.centers
    assert centers.shape == (3, X.shape[1])
    # each center should be close to the loc of normal distributions used to generate data
    for c in centers:
        assert any(np.allclose(c, np.array([i, i]), atol=0.05) for i in range(3))


def test_kpp_initialization_fit():
    np.random.seed(11)
    X = np.vstack([np.random.normal(loc=i, scale=0.01, size=(5, 2)) for i in range(3)])
    model = FuzzyCMeans(num_clusters=3, max_iter=100)
    np.random.seed(43)
    model.fit(X, initialization='k++')
    centers = model.centers
    assert centers.shape == (3, X.shape[1])
    for c in centers:
        assert any(np.allclose(c, np.array([i, i]), atol=0.05) for i in range(3))


def test_federated_initialization_requires_pre_set_centers_and_then_fits():
    np.random.seed(2)
    X = np.vstack([np.random.normal(loc=0.0, scale=0.1, size=(6, 2)),
                   np.random.normal(loc=1.0, scale=0.1, size=(6, 2))])

    model = FuzzyCMeans(num_clusters=2, max_iter=20)
    with pytest.raises(AssertionError):
        model.fit(X, initialization='federated')

    initial_centers = np.array([[0.0, 0.0], [1.0, 1.0]])
    model.set_centers(initial_centers.copy())
    assert np.allclose(model.centers, initial_centers)
    model.fit(X, initialization='federated')
    assert model.iterations > 0
    U = model.predict(X)
    assert U.shape == (X.shape[0], 2)
    assert np.allclose(U.sum(axis=1), 1.0, atol=1e-6)


def test_private_init_membership_and_update_centers_work():
    # Directly exercise private initializers and updaters via name-mangling
    np.random.seed(3)
    X = np.vstack([np.random.normal(loc=0.0, scale=0.1, size=(8, 2)),
                   np.random.normal(loc=1.0, scale=0.1, size=(8, 2))])

    model = FuzzyCMeans(num_clusters=2, max_iter=5)
    # initialize centers randomly (private)
    model._FuzzyCMeans__init_centers(X, 'random')
    assert model.centers is not None
    assert model.centers.shape == (2, X.shape[1])

    # initialize membership randomly (private)
    model._FuzzyCMeans__init_membership(X, 'random')
    U = model._FuzzyCMeans__U
    assert U is not None
    assert U.shape == (X.shape[0], 2)
    assert np.allclose(U.sum(axis=1), 1.0, atol=1e-8)

    # update centers based on membership (private)
    model._FuzzyCMeans__update_cluster_centers(X)
    c = model.centers
    assert c.shape == (2, X.shape[1])
    assert np.all(np.isfinite(c))


def test_predict_after_fit_consistent_membership():
    np.random.seed(5)
    X = np.vstack([np.random.normal(loc=0.0, scale=0.05, size=(7, 2)),
                   np.random.normal(loc=1.0, scale=0.05, size=(7, 2))])

    model = FuzzyCMeans(num_clusters=2, max_iter=50, tol=1e-6)
    np.random.seed(10)
    model.fit(X, initialization='random')
    U1 = model.predict(X)
    U2 = model.predict(X)  # repeated calls should be consistent
    assert np.allclose(U1, U2)
    assert U1.shape == (X.shape[0], 2)
    assert np.allclose(U1.sum(axis=1), 1.0, atol=1e-6)
