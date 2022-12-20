from utils import DataLoader, iso_scale, normalize, compute_quadratic_features
import numpy as np
from sklearn.preprocessing import scale
from sklearn.model_selection import ShuffleSplit, GridSearchCV
from quad_jax import QuadraticClassifier
from sklearn.svm import LinearSVC
import pandas as pd
import argparse

RANDOM_STATE = 0
TEST_SIZE = 0.2
C_SVM_PARAM = {"C": [0.0000001, 0.000001, 0.00001, 0.0001, 0.001, 0.01, 0.1, 1, 10]}
NUC_QUAD_RADIUS_PARAM = {"lmbda": [0.0001, 0.001, 0.01, 0.1, 1, 10, 50, 100, 500, 1000, 10000]}
CORES = 4
DATA_COPIES = CORES

parser = argparse.ArgumentParser()
parser.add_argument("--dataset", help="dataset name")
parser.add_argument("--n_runs", type=int, default=5, help="number of independent runs")
parser.add_argument("--n_cvfolds", type=int, default=4, help="cross val folds")
args = parser.parse_args()

dataset = DataLoader(args.dataset)
n_runs = args.n_runs
n_cvfolds = args.n_cvfolds

data_splits = ShuffleSplit(n_splits=n_runs, test_size=TEST_SIZE, random_state=RANDOM_STATE)
X, y = dataset.X, dataset.y

experimental_data = {'svm_train_acc': [],
                     'svm_test_acc': [],
                     'svm_compo_scaled_train_acc' : [],
                     'svm_compo_scaled_test_acc': [],
                     'svm_scaled_train_acc' : [],
                     'svm_scaled_test_acc': [],
                     'nuc_train_acc': [],
                     'nuc_test_acc': [],
                     'nuc_compo_scaled_train_acc' : [],
                     'nuc_compo_scaled_test_acc': [],
                     'nuc_scaled_train_acc' : [],
                     'nuc_scaled_test_acc': []
                     }
run = 1
for train_idx, test_idx in data_splits.split(X):

    X_train, y_train = X[train_idx], y[train_idx]
    X_test, y_test = X[test_idx], y[test_idx]

    svm = GridSearchCV(LinearSVC(max_iter=10000, fit_intercept=False), C_SVM_PARAM, cv=n_cvfolds, refit=True, n_jobs=CORES, pre_dispatch=DATA_COPIES)
    X_quad_train, X_quad_test = compute_quadratic_features(X_train, X_test, homogeneous=False)
    svm.fit(X_quad_train, y_train)
    svm_train_acc = svm.score(X_quad_train, y_train)
    svm_test_acc = svm.score(X_quad_test, y_test)

    nuc = QuadraticClassifier(dim=X.shape[1])
    nuc_grid_search = GridSearchCV(nuc, NUC_QUAD_RADIUS_PARAM, cv=n_cvfolds, n_jobs=CORES, pre_dispatch=DATA_COPIES)
    nuc_grid_search.fit(X_train, y_train, batch_size=X_train.shape[0])
    nuc.lmbda = nuc_grid_search.best_params_['lmbda']
    nuc.fit(X_train, y_train, batch_size=X_train.shape[0], plot=True, fname=args.dataset+"_plot_run_"+str(run))
    nuc_train_acc, nuc_test_acc = nuc.score(X_train, y_train), nuc.score(X_test, y_test)

    # COMPONENT WISE SCALING
    training_mean = np.mean(X_train, axis=0)
    training_component_stds = np.std(X_train, axis=0)
    X_train_scaled = scale(X_train)
    X_test_scaled = (X_test - training_mean)
    np.true_divide(X_test_scaled, training_component_stds, out=X_test_scaled, where=training_component_stds != 0)

    X_quad_train, X_quad_test = compute_quadratic_features(X_train_scaled, X_test_scaled, homogeneous=False)
    svm.fit(X_quad_train, y_train)
    svm_compo_scaled_train_acc, svm_compo_scaled_test_acc = svm.score(X_quad_train, y_train), svm.score(X_quad_test, y_test)


    nuc_grid_search.fit(X_train_scaled, y_train, batch_size=X_train.shape[0])
    nuc.lmbda = nuc_grid_search.best_params_['lmbda']
    nuc.fit(X_train_scaled, y_train, batch_size=X_train.shape[0], plot=True, fname=args.dataset+"_compo_scaled_plot_run"+str(run))
    nuc_compo_scaled_train_acc, nuc_compo_scaled_test_acc = nuc.score(X_train_scaled, y_train), nuc.score(X_test_scaled, y_test)


    # ISOTROPIC NORMALIZATION
    training_mean = np.mean(X_train, axis=0)
    X_train_scaled, training_covariance = iso_scale(scale(X_train, with_std=False))
    X_test_scaled = normalize(X_test - training_mean, training_covariance)

    X_quad_train, X_quad_test = compute_quadratic_features(X_train_scaled, X_test_scaled, homogeneous=False)
    svm.fit(X_quad_train, y_train)
    svm_scaled_train_acc, svm_scaled_test_acc = svm.score(X_quad_train, y_train), svm.score(X_quad_test, y_test)


    nuc_grid_search.fit(X_train_scaled, y_train, batch_size=X_train.shape[0])
    nuc.lmbda = nuc_grid_search.best_params_['lmbda']
    nuc.fit(X_train_scaled, y_train, batch_size=X_train.shape[0], plot=True, fname=args.dataset+"_scaled_plot_run"+str(run))
    nuc_scaled_train_acc, nuc_scaled_test_acc = nuc.score(X_train_scaled, y_train), nuc.score(X_test_scaled, y_test)


    #LOGGING
    experimental_data["svm_train_acc"].append(svm_train_acc)
    experimental_data["svm_test_acc"].append(svm_test_acc)
    experimental_data["svm_compo_scaled_train_acc"].append(svm_compo_scaled_train_acc)
    experimental_data["svm_compo_scaled_test_acc"].append(svm_compo_scaled_test_acc)
    experimental_data["svm_scaled_train_acc"].append(svm_scaled_train_acc)
    experimental_data["svm_scaled_test_acc"].append(svm_scaled_test_acc)
    experimental_data["nuc_train_acc"].append(nuc_train_acc)
    experimental_data["nuc_test_acc"].append(nuc_test_acc)
    experimental_data["nuc_compo_scaled_train_acc"].append(nuc_compo_scaled_train_acc)
    experimental_data["nuc_compo_scaled_test_acc"].append(nuc_compo_scaled_test_acc)
    experimental_data["nuc_scaled_train_acc"].append(nuc_scaled_train_acc)
    experimental_data["nuc_scaled_test_acc"].append(nuc_scaled_test_acc)

    run += 1

df = pd.DataFrame(experimental_data)
df.to_csv("../log/"+"squared_hinge_"+args.dataset+".csv")
