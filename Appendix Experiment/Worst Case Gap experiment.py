from quad_ascent import *
from utils import DataLoader, iso_scale, normalize, compute_quadratic_features
import numpy as np
import jax.numpy as jnp
from sklearn.preprocessing import scale
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
epochs = 25000

np.random.seed(23451*2**args.id)
A_true = np.random.randn(args.dim, args.dim)

def data_generation(d, isotropic=False):

    n_train = 5 * d * int(np.log(d))
    n_test = 5 * d * int(np.log(d))

    X = np.array([
            np.sqrt(d) * x / np.linalg.norm(x)
            for x in np.random.randn(n_train + n_test, d)
        ])

    if not isotropic:
        X = anisotropize(X, 0.999)

    X_train = X[:n_train, :]
    X_test = X[n_train:, :]


    X_train_scaled = X_train
    X_test_scaled = X_test

    y_train = np.random.binomial(1, 0.5, n_train)#np.sign(batch_classifier(A_true, X_train_scaled))
    y_test = np.random.binomial(1, 0.5, n_test)#np.sign(batch_classifier(A_true, X_test_scaled))

    print(int(np.log(d)), n_train, X_train.shape, y_train.shape)

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
        X_test, 
        y_test,
        n_epoch=epochs,
        plot=(args.id == 0),
        fname="worst nuc aniso" + str(d))

NUC_ANISO_MAX = batch_loss(nuc.A, X_train, y_train, X_test, y_test)

fro.fit(X_train,
        y_train, 
        X_test, 
        y_test,
        n_epoch=epochs,
        plot=(args.id == 0),
        fname="worst fro aniso" + str(d))

FRO_ANISO_MAX = batch_loss(fro.A, X_train, y_train, X_test, y_test)

X_train, y_train, X_test, y_test = data_generation(d, isotropic=True)

nuc.fit(X_train,
        y_train, 
        X_test, 
        y_test,
        n_epoch=epochs,
        plot=(args.id == 0),
        fname="worst nuc iso" + str(d))

NUC_ISO_MAX = batch_loss(nuc.A, X_train, y_train, X_test, y_test)

fro.fit(X_train,
        y_train, 
        X_test, 
        y_test,
        n_epoch=epochs,
        plot=(args.id == 0),
        fname="worst fro iso" + str(d))

FRO_ISO_MAX = batch_loss(fro.A, X_train, y_train, X_test, y_test)

data = {'nuc_iso': NUC_ISO_MAX,
        'nuc_aniso': NUC_ANISO_MAX,
        'fro_iso': FRO_ISO_MAX,
        'fro_aniso': FRO_ANISO_MAX}

df = pd.Series(data)
df.to_csv("../../log/" + "random_worst_basic_dim_" + str(args.dim) + "_run_"+str(args.id)+".csv")

