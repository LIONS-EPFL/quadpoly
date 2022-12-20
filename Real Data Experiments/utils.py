import numpy as np
from sklearn.datasets import load_svmlight_file
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
import pandas as pd
from tqdm import tqdm
import torch
from scipy.linalg import sqrtm, eigh


class DataLoader():
    def __init__(self, fname):
        data = load_svmlight_file("../datasets/" + fname)

        self.X = data[0].toarray()

        self.pos_label, self.neg_label = list(set(data[1]))

        def relabel(t):
            if t == self.pos_label:
                return 1
            else:
                return -1

        self.y = np.array([relabel(t) for t in data[1]])

        self.sparseX = data[0]

        self.X_quad = None
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None

    def split(self, test_size, random_seed):
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            self.X_quad, self.y, test_size=test_size, random_state=random_seed)

    def compute_quadratic_features(self, homogeneous=False):
        self.X_quad = []
        print("Computing quadratic features...")
        for x in tqdm(self.X):
            if not homogeneous:
                x = np.concatenate([x, [1]])
            self.X_quad.append(np.outer(x, x).reshape(1, -1)[0])


def find_best_parameters(estimator,
                         parameters,
                         X,
                         y,
                         n_folds_cv=5,
                         logfile="log"):
    cross_val = GridSearchCV(estimator, parameters, cv=n_folds_cv)
    cross_val.fit(X, y)
    results = pd.DataFrame(cross_val.cv_results_)
    print(
        "Best Cross Val Average score ", results.loc[
            results['rank_test_score'] == 1]['mean_test_score'].to_numpy()[0])
    results.to_csv("../log/" + logfile + ".csv")
    return cross_val.best_params_


def power_method(A, error=1e-4, maxiter=400):
    d = A.shape[0]
    if A.norm() < 0.001:
        return torch.zeros(d)
    u = torch.ones(d)
    u /= u.norm()
    uprev = u
    err = 1
    err_ = 1
    i = 0
    while err > error / d and err_ > error / d and i < maxiter:
        i += 1
        u = torch.matmul(A, u)
        u /= u.norm()
        err = (u - uprev).norm()
        err_ = (u + uprev).norm()
        uprev = u
    if i == maxiter:
        print("err not achieved")
    return u


def sample_spherical(npoints, ndim=3):
    vec = np.random.randn(ndim, npoints)
    vec *= np.sqrt(ndim) / np.linalg.norm(vec, axis=0)
    return vec.T


def inject_noise(y, noise):
    noise_ = 2 * np.random.binomial(1, noise, len(y)) - 1
    return np.multiply(y, noise_)


def simplex_projection(s):
    """Projection onto the unit simplex."""
    if np.sum(s) <= 1 and np.alltrue(s >= 0):
        return s
    # Code taken from https://gist.github.com/daien/1272551
    # get the array of cumulative sums of a sorted (decreasing) copy of v
    u = np.sort(s)[::-1]
    cssv = np.cumsum(u)
    # get the number of > 0 components of the optimal solution
    try:
        rho = np.nonzero(u * np.arange(1, len(u) + 1) > (cssv - 1))[0][-1]
    except Exception:
        rho = 0
    # compute the Lagrange multiplier associated to the simplex constraint
    theta = (cssv[rho] - 1) / (rho + 1.0)
    # compute the projection by thresholding v using theta
    return np.maximum(s - theta, 0)


def nuclear_projection(A, lam):
    """Projection onto nuclear norm ball."""
    U, s, V = np.linalg.svd(A, full_matrices=False)
    s = lam * simplex_projection(s / lam)
    return U.dot(np.diag(s).dot(V))


def frobenius_projection(A, lam):
    fro = np.linalg.norm(A, ord='fro')
    return lam * A / max(lam, fro)


def rank_constraint(A, r):
    U, s, V = np.linalg.svd(A, full_matrices=False)
    s[r:] = 0
    return U.dot(np.diag(s).dot(V))


def compute_quadratic_features(X_train, X_test, homogeneous=True):
    X_quad_train = []
    for x in X_train:
        if not homogeneous:
            x = np.concatenate([x, [1]])
        X_quad_train.append(np.outer(x, x).reshape(1, -1)[0])
    X_quad_test = []
    for x in X_test:
        if not homogeneous:
            x = np.concatenate([x, [1]])
        X_quad_test.append(np.outer(x, x).reshape(1, -1)[0])
    return np.array(X_quad_train), np.array(X_quad_test)


def matrix_sqrt(M):
    w, v = eigh(M, check_finite=True)
    w = np.maximum(w, 0)
    return (v * np.sqrt(w)).dot(v.conj().T)


def iso_scale(X):
    X_quad, _ = compute_quadratic_features(X, [])
    empirical_covariance = np.mean(X_quad, axis=0)
    empirical_covariance = empirical_covariance.reshape(
        int(np.sqrt(max(empirical_covariance.shape))), -1)

    isotropicX = []
    sqrt_inverse_empirical_covariance = matrix_sqrt(
        np.linalg.pinv(empirical_covariance, hermitian=True))
    for x in X:
        x_iso = sqrt_inverse_empirical_covariance @ x
        isotropicX.append(x_iso)
    return np.array(isotropicX), empirical_covariance


def normalize(X, sigma):
    normalized = []
    sqrt_inv_sigma = matrix_sqrt(np.linalg.pinv(sigma, hermitian=True))
    for x in X:
        x_iso = sqrt_inv_sigma @ x
        normalized.append(x_iso)
    return np.array(normalized)


def anisotropize(X, s):
    d = len(X[0])
    v = np.array([1 / (i + 1)**s for i in range(d)])
    v /= np.linalg.norm(v)
    v *= np.sqrt(d)
    X_ = np.multiply(X, v)
    return X_
