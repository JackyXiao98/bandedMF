from functools import reduce
import numpy as np
from scipy import optimize
from scipy import sparse
from scipy.sparse.linalg import spsolve_triangular
from scipy.linalg.lapack import dpotrf, dpotri
from utils import *


class BandedConvex:
    def __init__(self, n, b=2):
        self.n = n
        self._mask = np.tri(n, dtype=bool, k=-1)
        self._params = np.zeros(n * (n - 1) // 2)
        # self.X = np.zeros((n, n))
        self.X = np.eye(n)
        self._band_mask = np.abs(np.subtract.outer(np.arange(n), np.arange(n))) < b
        self.b = b

    def strategy_pmat(self):
        tri = np.zeros((self.n, self.n))
        tri[self._mask] = self._params
        X = np.eye(self.n) + tri + tri.T
        self.X = X
        # A = np.linalg.cholesky(X).T
        return X

    def strategy_matrix(self):
        X = self.X
        A = np.linalg.cholesky(X).T
        self.A = A
        return A

    def _set_workload(self, W):
        # self.V = W.gram().dense_matrix().astype(float)
        self.V = W.T @ W
        self.W = W

    def _loss_and_grad(self, params):
        V = self.V
        X = self.X
        X.fill(0)
        # X = np.zeros((self.n,self.n))
        X[self._mask] = params
        X += X.T
        np.fill_diagonal(X, 1)

        zz, info0 = dpotrf(X, False, False)
        iX, info1 = dpotri(zz)
        iX = np.triu(iX) + np.triu(iX, k=1).T

        # if not psd, return a large loss
        if info0 != 0 or info1 != 0:
            # print('checkpt')
            return self._loss * 100, np.zeros_like(params)

        loss = np.sum(iX * V)
        G = -iX @ V @ iX

        # set gradient to 0
        G[~self._band_mask] = 0

        g = G[self._mask] + G.T[self._mask]

        self._loss = loss
        # print(np.sqrt(loss / self.W.shape[0]))
        return loss, g  # G.flatten()

    def optimize(self, W):
        self._set_workload(W)

        # initialization
        X = np.eye(self.n)
        x = X[self._mask]

        # x = np.eye(self.n).flatten()
        # bnds = [(1,1) if x[i] == 1 else (None, None) for i in range(x.size)]
        # x = self._params

        opts = {'maxcor': 1}
        res = optimize.minimize(self._loss_and_grad, x, jac=True, method='L-BFGS-B', options=opts)
        self._params = res.x
        # print(res)
        return res.fun


if __name__ == "__main__":
    n = 100
    b = n
    np.set_printoptions(precision=3)
    W = np.tril(np.ones((n, n)), k=0)

    # upper = 10
    # diag = np.arange(0.0, upper, upper/n) + 1.0
    # diag = [1] * 40 + [upper] * 10
    # diag = np.array(diag)
    diag = np.ones(n)
    W = np.diag(diag) @ W

    temp = BandedConvex(n, b)
    fun = temp.optimize(W)
    X = temp.strategy_pmat()
    rot = rotation_matrix(n)
    X_rot = rot @ X @ rot
    A_rot = np.linalg.cholesky(X_rot).T
    A = rot @ A_rot @ rot
    # print("strategy matrix:\n", A)
    # print("pcost matrix:\n", X)
    L = W @ np.linalg.pinv(A)
    L = np.diag(1.0/diag) @ L
    var = np.diag(L @ L.T)
    print("sum of var: ", np.sum(var))
    print("max of var: ", np.max(var))

    # A_avg = average_matrix(A, b, k=10)
    # p_mat_avg = A_avg.T @ A_avg
    # p_cost_avg = np.max(np.diag(p_mat_avg))
    # C_avg = W @ np.linalg.pinv(A_avg)
    # var_avg = np.trace(C_avg @ C_avg.T)

