from sklearn.base import BaseEstimator
from jax import grad, jit, jacfwd, vmap, partial, hessian
import jax.numpy as jnp
import jax.scipy as jsp
import jax.lax as lax
import numpy as np
from jax import jacfwd
import matplotlib.pyplot as plt
from jax.ops import index, index_add, index_update


class QuadraticClassifier(BaseEstimator):
    def __init__(self, dim, lmbda=1, norm="nuc"):
        super().__init__()
        self.lmbda = lmbda
        self.dim = dim
        self.norm = norm

        # xAx + bx + c
        self.A = jnp.zeros((self.dim, self.dim), dtype=jnp.float32)
        self.b = jnp.zeros(self.dim, dtype=jnp.float32)
        self.c = 0.0

        

    def projected_GD(self, grad_A, step_size):

        step_A = step_size

        # Gradient step
        self.A -= step_A * grad_A

        # project A
        if self.norm == "nuc":
            self.A = nuclear_project(self.A, self.lmbda, self.dim)
        if self.norm == "fro":
            self.A = frobenius_project(self.A, self.lmbda)

    
    def fit(self, X_train, y_train, X_test, y_test, n_epoch=1000, plot=False, fname=None):

        
        batch_grad = jit(grad(lambda A: batch_loss(A, X_train, y_train, X_test, y_test)))
        b_l = jit(lambda A : batch_loss(A, X_train, y_train, X_test, y_test))

        train_losses = np.zeros((n_epoch, 1))
        A_best = self.A
        f_best = float('inf')

        L = np.average([np.linalg.norm(x)**4 for x in X_train])

        A_prev = jnp.array(self.A, copy=True)

        m_A = 0

        for i in range(n_epoch):

            grad_A = batch_grad(A_prev)
            
            step_A = 1.0 / L

            # param_updates
            self.projected_GD(grad_A, step_A)

            f_curr = b_l(self.A)

            if plot:
                train_losses[i, 0] = -f_curr

            m_A = (1 * self.A - A_prev)
            A_prev = index_update(A_prev, index[:, :], self.A)           

            if L * (np.linalg.norm(m_A)**2) < 1e-10:
                print("JUST HIT TOLERANCE")
                break
            if f_curr < f_best:
                A_best = index_update(A_best, index[:, :], self.A)
                f_best = f_curr

        self.A = A_best
        if plot:
            plt.yscale('log')
            plt.plot(train_losses[:, 0], label="iterates")
            plt.legend()
            plt.savefig("old_figs/{}.png".format(fname), format='png')
            plt.close()

    def predict(self, X):
        return jnp.sign(batch_classifier(self.A, self.b, self.c, X))

    def score(self, X, y):
        preds = self.predict(X)
        return 1 - jnp.sum(jnp.abs(preds - y)) / (2 * len(X))


@jit
def classifier(A, x):
    return jnp.dot(x, jnp.dot(A, x))


batch_classifier = jit(vmap(classifier, in_axes=(None, 0)))


@jit
def hinge_loss(y, x):
    return jnp.maximum(0, 1 - y * x)


@jit
def smoothed_hinge_loss(y, x):
    return jnp.where(y * x <= 0, 0.5 - y * x,
                     jnp.where(y * x < 1, 0.5 * (1 - y * x)**2, 0))


@jit
def loss(A, x_train, y_train, x_test, y_test):
    return smoothed_hinge_loss(y_test, classifier(A, x_test)) - smoothed_hinge_loss(y_train, classifier(A, x_train))


@jit
def batch_loss(A, x_train, y_train, x_test, y_test):
    train = batch_classifier(A, x_train)
    test = batch_classifier(A, x_test)
    return -(jnp.mean(smoothed_hinge_loss(y_test, test)) - jnp.mean(smoothed_hinge_loss(y_train, train))) 



@partial(jit, static_argnums=(1, 2))
def project(v, radius, dim):
    mu = lax.sort(v)
    cumul_sum = jnp.divide(
        lax.cumsum(mu, reverse=True) - radius, jnp.arange(dim, 0, -1))
    rho = jnp.amin(jnp.where(mu > cumul_sum, jnp.arange(dim), dim))
    theta = cumul_sum[rho]
    return jnp.maximum(v - theta, 0)

@jit
def svd(mat):
    return jsp.linalg.svd(mat, full_matrices=False)


@partial(jit, static_argnums=(1, 2))
def nuclear_project(mat, radius, dim):
    U, s, Vt = svd(mat)
    return jnp.where(jnp.sum(s) > radius, jnp.dot(U * project(s, radius, dim), Vt), mat)


@partial(jit, static_argnums=1)
def frobenius_project(mat, radius):
    norm_mat = jnp.linalg.norm(mat, ord='fro')
    return jnp.where(norm_mat > radius, radius / norm_mat * mat, mat)

