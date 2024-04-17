# -*- coding: utf-8 -*-
"""
MIT License

Copyright (c) 2022 cmla-psu/Yingtai Xiao

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
"""


import argparse
import numpy as np
from functools import reduce
from scipy import optimize
from scipy import sparse
from scipy.sparse.linalg import spsolve_triangular
from scipy.linalg.lapack import dpotrf, dpotri


"""
Todo tasks

* [Done] If not PSD, set a large loss, set the gradient to 0.
* [Done] Iterative approximation, increase t for each round, set up a stopping condition
* Reduce the number of parameters by half, use a mask, see HDMM solver.
* Find proper initialization.
* Replace the inverse function.
* [Done] Choose different variance bounds.
* Banded matrix, use masks.


"""


def configuration():
    """
    Return configuration parameters.

    Returns
    -------
    args : parameters.

    """
    parser = argparse.ArgumentParser(description='Matrix Query')

    parser.add_argument('--maxiter', default=1000000, help='total iteration')
    parser.add_argument('--maxitercg', default=5,
                        help='maximum iteration for conjugate gradient method')
    parser.add_argument('--maxiterls', default=50,
                        help='maximum iteration for finding a step size')
    parser.add_argument('--theta', default=1e-10,
                        help='determine when to stop conjugate gradient method'
                        ' smaller theta makes the step direction more accurate'
                        ' but it takes more time')
    parser.add_argument('--beta', default=0.5, help='step size decrement')
    parser.add_argument('--sigma', default=1e-2,
                        help='determine how much decrement in '
                        'objective function is sufficient')
    parser.add_argument('--NTTOL', default=1e-3,
                        help='determine when to update self.param_t, '
                        'smaller NTTOL makes the result more accurate '
                        'and the convergence slower')
    parser.add_argument('--TOL', default=1e-3,
                        help='determine when to stop the whole program')
    parser.add_argument('--MU', default=2, help='increment for '
                        'barrier approximation parameter self.param_t')
    parser.add_argument('--init_mat', default='id_index',
                        help='id_index method is sufficient as an initialization.')
    parser.add_argument('--basis', default='work',
                        help='id: id mat; work: work mat')
    return parser.parse_args()


def workload(n, dtype=np.float32):
    """
    Create the upper triangular matrix with elements of 1.

    Return as the basis matrix.
    """
    mat_w = np.triu(np.ones([n, n]))
    return mat_w


def is_pos_def(A):
    """
    Check positive definiteness.

    Return true if psd.
    """
    # first check symmetry
    if np.allclose(A, A.T, 1e-5, 1e-8):
        # whether cholesky decomposition is successful
        try:
            np.linalg.cholesky(A)
            return True
        except np.linalg.LinAlgError:
            return False
    else:
        return False


def hb_strategy_matrix(n, k):
    """
    Strategy matrix of HB method.

    here requires n is a square number so that n = k*k
    special case : if n=2, return a given matrix

    Parameters
    ----------
    n : the number of nodes
    k : the branching factor
    """
    if n == 2:
        Tree = np.array([[1, 1], [1, 0], [0, 1]])
    else:
        m = 1 + k + n
        Tree = np.zeros([m, n])
        Tree[0, :] = 1
        for i in range(k):
            Tree[i+1, k*i: k*i+k] = 1
        Tree[k+1:, :] = np.eye(n)
    return Tree


def hb_variance(W, A, s):
    """
    Calculate the maximum variance for a single query in workload W.

    using HB method with privacy cost at most s.

    Parameters
    ----------
    A is the hb tree structure
    s is the privacy cost
    """
    m, n = A.shape
    mI = np.eye(m)
    pA = np.linalg.pinv(A)
    sigma = np.sqrt(3/s)
    Var = W @ pA
    BXB = Var @ (sigma**2*mI) @ Var.T
    a = np.diag(BXB)
    return a


def gm_variance(W, A, s):
    """
    Calculate the maximum variance for a single query in workload W.

    Parameters
    ----------
    s is the privacy cost
    """
    m, n = A.shape
    mI = np.eye(m)
    # pseudoinverse of matrix A
    pA = np.linalg.inv(A)
    sigma = np.sqrt(1/s)
    Var = W @ pA
    BXB = Var @ (sigma**2*mI) @ Var.T
    a = np.diag(BXB)
    return a


class matrix_query:
    """Class for matrix query optimization."""

    def __init__(self, args, mat_basis=None, mat_index=None, var_bound=None):
        """
        Get Inputs.

        Parameters
        ----------
        args : TYPE
            Configuration parameters.
        mat_basis : np.ndarray
            Basis matrix. The default is None.
        mat_index : np.ndarray
            Index matrix. The default is None.
        var_bound : np.ndarray
            Variance bound. The default is None.

        Returns
        -------
        None.

        """
        self.size_m, self.size_n = mat_index.shape
        self.mat_basis = mat_basis
        self.mat_index = mat_index
        self.mat_work = self.mat_index @ self.mat_basis
        self.var_bound = var_bound
        self.args = args
        self.initialization()

    def initialization(self):
        """
        Initialize covariance matrix and privacy cost.

        Returns
        -------
        None.

        """
        self.mat_id = np.eye(self.size_n)
        self.param_t = 1
        self.param_k = 1
        if self.args.init_mat == 'id_index':
            diag = np.diag(self.mat_index @ self.mat_index.T)
            sigma = np.min(self.var_bound/diag)
            self.cov = self.mat_id*self.size_n*sigma
        self.invcov = np.linalg.solve(self.cov, self.mat_id)
        self.f_var = self.func_var()
        self.f_pcost = self.func_pcost()

    def func_var(self):
        """
        Inequality constraint function for variance.

        Parameters
        ----------
        self.var_bound : variance bound
        self.mat_index : the index matrix
        self.cov : the co-variance matrix
        """
        # d = np.diag(self.mat_index @ self.cov @ self.mat_index.T)
        vec_d = ((self.mat_index @ self.cov) * self.mat_index).sum(axis=1)
        # vec_d = np.diag(self.cov)
        return vec_d / self.var_bound

    def func_pcost(self):
        """
        Inequality constraint function for privacy cost.

        Parameters
        ----------
        B : the basis matrix
        self.invcov : the inverse of the co-variance matrix X
        """
        # d = np.diag(self.mat_basis.T @ self.invcov @ self.mat_basis)
        vec_d = ((self.mat_basis.T @ self.invcov) * self.mat_basis.T).sum(
            axis=1)
        # vec_d = np.diag(self.invcov)
        # vec_d = np.triu(self.invcov.cumsum(axis=1)).sum(axis=0)
        return vec_d

    def obj(self):
        """
        Objective function.

        Parameters
        ----------
        X : co-variance matrix
        self.param_t : privacy cost approximation parameter
        self.param_k : variance approximation parameter
        c : variance bound, self.mat_index: index matrix, B: basis matrix
        """
        const_t = self.param_t*np.max(self.f_pcost)
        const_k = self.param_k*np.max(self.f_var)
        log_sum_t = np.log(np.sum(np.exp(self.param_t*self.f_pcost - const_t)))
        log_sum_k = np.log(np.sum(np.exp(self.param_k*self.f_var - const_k)))
        f_obj = const_t + log_sum_t + const_k + log_sum_k
        return f_obj / self.param_t

    def derivative(self):
        """Calculate derivatives."""
        const_k = self.param_k * np.max(self.f_var)
        exp_k = np.exp(self.param_k*self.f_var - const_k)
        self.g_var = (exp_k/self.var_bound*self.mat_index.T) @ self.mat_index

        const_t = np.max(self.param_t * self.f_pcost)
        self.mat_bix = self.invcov.T @ self.mat_basis
        exp_t = np.exp(self.param_t*self.f_pcost-const_t)
        self.g_pcost = -(exp_t*self.mat_bix) @ self.mat_bix.T

        coef_k = self.param_k/np.sum(exp_k)
        coef_t = self.param_t/np.sum(exp_t)
        grad_k = self.g_var * coef_k
        grad_t = self.g_pcost * coef_t

        grad = grad_k + grad_t
        vec_grad = np.reshape(grad, [-1], 'F')
        return vec_grad

    def _loss_and_grad(self, params):
        self.cov = np.reshape(params, [self.size_n, self.size_n], 'F')
        if not is_pos_def(self.cov):
            self.cov = (self.cov + self.cov.T) / 2.0 + self.mat_id
            self.invcov = np.linalg.solve(self.cov, self.mat_id)
            self.f_var = self.func_var()
            self.f_pcost = self.func_pcost()
            loss = self.obj()
            g = self.derivative()
            return loss * 100, np.zeros_like(g)

        self.invcov = np.linalg.solve(self.cov, self.mat_id)
        self.f_var = self.func_var()
        self.f_pcost = self.func_pcost()

        loss = self.obj()
        g = self.derivative()
        return loss, g  # G.flatten()

    def optimize(self):
        # initialization
        # x = np.reshape(self.cov, [-1], 'F')

        opts = {'maxcor': 1}

        for iters in range(100):
            x = np.reshape(self.cov, [-1], 'F')
            res = optimize.minimize(self._loss_and_grad, x, jac=True, method='L-BFGS-B', options=opts)
            self._params = res.x

            gap = (self.size_m + self.size_n) / self.param_t
            if gap < self.args.TOL:
                break
            self.param_t = self.args.MU * self.param_t
            self.param_k = self.args.MU * self.param_t
            print('update t: {0}'.format(self.param_t))
            # print(self.cov)

        return res.fun


if __name__ == "__main__":
    np.set_printoptions(precision=3)
    np.random.seed(0)
    k = 20
    # work = np.eye(k)
    work = np.tril(np.ones([k, k]))
    param_m, param_n = work.shape
    bound = np.ones(param_m)*1
    # upper = 3
    # diag = np.arange(0.0, upper, upper/k) + 1.0
    # bound = np.array(diag)[::-1]

    args = configuration()
    args.init_mat = 'id_index'

    args.maxitercg = 5
    args.theta = 1e-8
    args.sigma = 1e-8
    args.NTTOL = 1e-5
    args.TOL = 1e-5

    index = work
    basis = np.eye(param_n)

    mat_opt = matrix_query(args, basis, index, bound)
    mat_opt.optimize()
    # mat_cov = mat_opt.cov/np.max(mat_opt.f_var)

    pmat_CA = mat_opt.invcov
    # ensure that the privacy cost is 1
    pmat_CA = pmat_CA / pmat_CA[0, 0]
    cov = np.linalg.inv(pmat_CA)
    B_inv = np.linalg.cholesky(cov)
    B = np.linalg.inv(B_inv)

    L = work @ B_inv
    var = np.diag(L @ L.T)
    print("sum of var: ", np.sum(var))
    print("max of var: ", np.max(var))
