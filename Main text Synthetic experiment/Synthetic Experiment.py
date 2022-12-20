from utils import DataLoader, iso_scale, normalize, compute_quadratic_features
import numpy as np
import jax.numpy as jnp
from sklearn.preprocessing import scale
from sklearn.model_selection import ShuffleSplit, GridSearchCV
from quad_jax import QuadraticClassifier, batch_loss, batch_classifier
from sklearn.svm import LinearSVC
import pandas as pd
import matplotlib.pyplot as plt
from utils import *
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--dim", type=int, default=10, help="Dimension")
parser.add_argument("--id", type=int, default=0, help="Dimension")
args = parser.parse_args()

lmbda = 1
epochs = 10000
noise = 0.001
isotropy = 2

np.random.seed(23451*2**args.id)
d_zero = int(np.floor(args.dim**0.5))
u, v = np.random.randn(2, d_zero)
A = np.outer(u, v)
A_true = A/np.linalg.norm(A, ord='nuc')


data = {
    'anisotropic_nuc_train': 0,
    'anisotropic_fro_train': 0,
    'isotropic_nuc_train': 0,
    'isotropic_fro_train': 0,
    'anisotropic_nuc_test': 0,
    'anisotropic_fro_test': 0,
    'isotropic_nuc_test': 0,
    'isotropic_fro_test': 0
}


def generate(n, d, eta, r):
    d_zero = int(np.floor(d**eta))
    O = np.linalg.svd(np.random.randn(d, d))[0]

    U = O[:, :d_zero]
    U_bot = O[:, d_zero:]

    r1 = r * np.sqrt(d_zero)
    Unif_1 = [r1 * x / np.linalg.norm(x) for x in np.random.randn(n, d_zero)]

    r2 = np.sqrt(d - d_zero)
    Unif_2 = [
        r2 * x / np.linalg.norm(x) for x in np.random.randn(n, d - d_zero)
    ]

    X = [
        np.dot(U, z_1) + np.dot(U_bot, z_2)
        for (z_1, z_2) in zip(Unif_1, Unif_2)
    ]

    return np.array(X), U



def data_generation(d, isotropic=False):

    n_train = 5 * d * int(np.log(d))
    n_test = 5 * d * int(np.log(d))

    X = np.array([
            np.sqrt(d) * x / np.linalg.norm(x)
            for x in np.random.randn(n_train + n_test, d)
        ])
    U = None

    if not isotropic:
        X, U = generate(n_train + n_test, d, 0.5, d**(isotropy))
    else:
        X, U = generate(n_train + n_test, d, 0.5, 1)

    X_train = X[:n_train, :]
    X_test = X[n_train:, :]

    training_mean = np.mean(X_train, axis=0)
    training_component_stds = np.std(X_train, axis=0)
    X_train_scaled = scale(X_train)
    X_test_scaled = (X_test - training_mean) / training_component_stds

    y_train = inject_noise(
        np.sign(batch_classifier(U@A_true@U.T, np.zeros(d), 0, X_train_scaled)),
        noise)
    y_test = inject_noise(
        np.sign(batch_classifier(U@A_true@U.T, np.zeros(d), 0, X_test_scaled)),
        noise)

    sigma_train = np.average([np.outer(x, x) for x in X_train_scaled], axis=0)
    sigma_test = np.average([np.outer(x, x) for x in X_test_scaled], axis=0)

    train_dim = np.trace(sigma_train) / np.linalg.norm(sigma_train, ord=2)
    test_dim = np.trace(sigma_test) / np.linalg.norm(sigma_test, ord=2)

    print("INTRINSICS ARE :", train_dim, test_dim, d)

    return X_train_scaled, y_train, X_test_scaled, y_test


d = args.dim
nuc = QuadraticClassifier(dim=d, lmbda=lmbda, norm='nuc')
fro = QuadraticClassifier(dim=d, lmbda=lmbda, norm='fro')

X_train, y_train, X_test, y_test = data_generation(d, isotropic=False)


nuc.fit(X_train,
        y_train,
        n_epoch=epochs,
        batch_size=len(X_train),
        plot=(args.id == 0),
        fname="SYNTH NUC LOW" + str(d))
fro.fit(X_train,
        y_train,
        n_epoch=epochs,
        batch_size=len(X_train),
        plot=(args.id == 0),
        fname="SYNTH FRO LOW" + str(d))

nuc_train_loss = batch_loss(nuc.A, nuc.b, nuc.c, X_train, y_train)
nuc_test_loss = batch_loss(nuc.A, nuc.b, nuc.c, X_test, y_test)
fro_train_loss = batch_loss(fro.A, fro.b, fro.c, X_train, y_train)
fro_test_loss = batch_loss(fro.A, fro.b, fro.c, X_test, y_test)

data['anisotropic_nuc_train'] = nuc_train_loss
data['anisotropic_fro_train']= fro_train_loss
data['anisotropic_nuc_test'] = nuc_test_loss
data['anisotropic_fro_test'] = fro_test_loss

X_train, y_train, X_test, y_test = data_generation(d, isotropic=True)

nuc.fit(X_train,
        y_train,
        n_epoch=epochs,
        batch_size=len(X_train),
        plot=(args.id == 0),
        fname="SYNTH NUC HIGH" + str(d))
fro.fit(X_train,
        y_train,
        n_epoch=epochs,
        batch_size=len(X_train),
        plot=(args.id == 0),
        fname="SYNTH FRO HIGH" + str(d))

nuc_train_loss = batch_loss(nuc.A, nuc.b, nuc.c, X_train, y_train)
nuc_test_loss = batch_loss(nuc.A, nuc.b, nuc.c, X_test, y_test)
fro_train_loss = batch_loss(fro.A, fro.b, fro.c, X_train, y_train)
fro_test_loss = batch_loss(fro.A, fro.b, fro.c, X_test, y_test)

data['isotropic_nuc_train'] = nuc_train_loss
data['isotropic_fro_train'] = fro_train_loss
data['isotropic_nuc_test'] = nuc_test_loss
data['isotropic_fro_test'] = fro_test_loss

df = pd.Series(data)
df.to_csv("../log/" + "old_synth_dim" + str(args.dim) + "run_"+str(args.id)+".csv")

print("DONEZOO")
