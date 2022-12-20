from sklearn.base import BaseEstimator
from jax import grad, jit, jacfwd, vmap, partial, hessian
import jax.numpy as jnp
import jax.scipy as jsp
import jax.lax as lax
import numpy as np
from jax import jacfwd
import matplotlib.pyplot as plt


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

        

    def projected_GD(self, grad_A, grad_b, grad_c, step_sizes):

        step_A, step_b, step_c = step_sizes

        # Gradient step
        self.A -= step_A * grad_A
        self.b -= step_b * grad_b
        self.c -= step_c * grad_c

        # project A
        if self.norm == "nuc":
            self.A = nuclear_project(self.A, self.lmbda, self.dim)
        if self.norm == "fro":
            self.A = frobenius_project(self.A, self.lmbda)

    def fit(self, X, y, n_epoch=1000, batch_size=10, plot=False, fname=None):

        n = X.shape[0]
        batch_grad = jit(grad(batch_loss, argnums=(0, 1, 2)))

        train_losses = np.zeros((n_epoch * (n // batch_size), 1))
        A_best = self.A
        b_best = self.b
        c_best = self.c
        f_best = float('inf')
        f_k = -1

        self.grad_acc_A = 0.0001
        self.grad_acc_b = 0.0001
        self.grad_acc_c = 0.0001

        L = np.average([np.linalg.norm(x)**4 for x in X])
        done = False

        A_prev = jnp.array(self.A, copy=True)
        b_prev = jnp.array(self.b, copy=True)
        c_prev = jnp.array(self.c, copy=True)

        A_prev_prev = jnp.array(self.A, copy=True)
        b_prev_prev = jnp.array(self.b, copy=True)
        c_prev_prev = jnp.array(self.c, copy=True)  



        for i in range(n_epoch):
            for k in range(n // batch_size):

                # batch
                inputs, labels = (
                    X[k * batch_size:(k + 1) * batch_size],
                    y[k * batch_size:(k + 1) * batch_size],
                )

                t = i+1
                v_A = A_prev + (t-2)/(t+1)*(A_prev - A_prev_prev)
                v_b = b_prev + (t-2)/(t+1)*(b_prev - b_prev_prev)
                v_c = c_prev + (t-2)/(t+1)*(c_prev - c_prev_prev)

                grad_A, grad_b, grad_c = batch_grad(v_A, v_b, v_c,
                                                    inputs, labels)

                #Ada_grad stepsizes

                # self.grad_acc_A += grad_A**2
                # self.grad_acc_b += grad_b**2
                # self.grad_acc_c += grad_c**2

                step_A = 1.0/L#jnp.sqrt(self.grad_acc_A)
                step_b = 1.0/L#jnp.sqrt(self.grad_acc_b)
                step_c = 1.0/L#jnp.sqrt(self.grad_acc_c)

                
                
                f_k = batch_loss(self.A, self.b, self.c, inputs, labels)

                # param_updates
                self.projected_GD(grad_A, grad_b, grad_c,
                                  (step_A, step_b, step_c))
                if L*(jnp.linalg.norm(self.A - A_prev)**2 + jnp.linalg.norm(self.b - b_prev)**2 + (self.c - c_prev)**2) < 1e-10:
                    done = True
                    print("JUST HIT TOLERANCE")
                    break

                A_prev, A_prev_prev = jnp.array(self.A, copy=True), jnp.array(A_prev, copy=True)
                b_prev, b_prev_prev = jnp.array(self.b, copy=True), jnp.array(b_prev, copy=True)
                c_prev, c_prev_prev = jnp.array(self.c, copy=True), jnp.array(c_prev, copy=True)

                #storing best so far
                f_k = batch_loss(self.A, self.b, self.c, inputs, labels)
                if f_k < f_best:
                    A_best = jnp.array(self.A, copy=True)
                    b_best = jnp.array(self.b, copy=True)
                    c_best = jnp.array(self.c, copy=True)
                    f_best = f_k

                if plot:
                    train_losses[i * (n // batch_size) + k,
                                 0] = batch_loss(self.A, self.b, self.c, X, y)
            if done:
                break
            
        self.A = A_best
        self.b = b_best
        self.c = c_best
        

        if plot:
            plt.yscale('log')
            plt.plot(train_losses[:, 0], label="iterates")
            plt.legend()
            plt.savefig("figs/{}.png".format(fname), format='png')
            plt.close()

    def predict(self, X):
        return jnp.sign(batch_classifier(self.A, self.b, self.c, X))

    def score(self, X, y):
        preds = self.predict(X)
        return 1 - jnp.sum(jnp.abs(preds - y)) / (2 * len(X))


@jit
def classifier(A, b, c, x):
    return jnp.dot(x, jnp.dot(A, x)) + jnp.dot(b, x) + c


batch_classifier = jit(vmap(classifier, in_axes=(None, None, None, 0)))


@jit
def hinge_loss(y, x):
    return jnp.maximum(0, 1 - y * x)


@jit
def smoothed_hinge_loss(y, x):
    return jnp.where(y * x <= 0, 0.5 - y * x,
                     jnp.where(y * x < 1, 0.5 * (1 - y * x)**2, 0))


@jit
def loss(A, b, c, x, y):
    return smoothed_hinge_loss(y, classifier(A, b, c, x))


@jit
def batch_loss(A, b, c, x, y):
    preds = batch_classifier(A, b, c, x)
    return jnp.mean(smoothed_hinge_loss(y, preds))


@jit
def gradient(A, b, c, x, y):
    return grad(loss, argnums=(0, 1, 2))(A, b, c, x, y)


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
    s = jnp.where(jnp.sum(s) > radius, project(s, radius, dim), s)
    return jnp.dot(U, jnp.multiply(s, Vt))


@partial(jit, static_argnums=1)
def frobenius_project(mat, radius):
    norm_mat = jnp.linalg.norm(mat, ord='fro')
    return jnp.where(norm_mat > radius, radius / norm_mat * mat, mat)
