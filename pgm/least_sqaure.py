import numpy as np
import cvxpy as cp
from functools import reduce


def subtract_matrix(k):
    """Return Subtraction matrix Sub_k."""
    mat = np.zeros([k-1, k])
    for i in range(k-1):
        mat[i, i] = 1
        mat[i, i+1] = -1
    return mat


np.random.seed(23)
np.set_printoptions(precision=3)
# vec_x = np.array([0, 2, 0, 0, 2, 0, 0, 0, 2, 0, 0, 1])
vec_x = np.array([0, 2, 20, 0, 2, 0, 0, 0, 2, 50, 0, 1])
# vec_x = np.random.randint(1, 30, 12)
# vec_x = np.ones(12) * 10
ls_ab = [np.eye(2), np.eye(2), np.ones([1, 3])]
ls_bc = [np.ones([1, 2]), np.eye(2), np.eye(3)]
ls_ac = [np.eye(2), np.ones([1, 2]), np.eye(3)]

Mab = reduce(np.kron, ls_ab)
Mbc = reduce(np.kron, ls_bc)
Mac = reduce(np.kron, ls_ac)

yab = Mab @ vec_x + np.random.normal(size=Mab.shape[0])
ybc = Mbc @ vec_x + np.random.normal(size=Mbc.shape[0])

x = cp.Variable(vec_x.shape, nonneg=True)
objective = cp.Minimize(cp.norm(Mab @ x - yab, 2) + cp.norm(Mbc @ x - ybc, 2))
problem = cp.Problem(objective)
problem.solve(solver=cp.ECOS)

print("The optimal value for x is:", x.value)
print("The L1 error for x is: ", np.linalg.norm(x.value-vec_x, 1))
diff = Mac @ (x.value - vec_x)
error = np.linalg.norm(diff, 1)
print("The L1 error for non-negative lsq is: ", error)

ls_sum = [np.ones([1, 2]), np.ones([1, 2]), np.ones([1, 3])]
ls_res_a = [np.array([1, -1]), np.ones([1, 2]), np.ones([1, 3])]
ls_res_b = [np.ones([1, 2]), subtract_matrix(2), np.ones([1, 3])]
ls_res_c = [np.ones([1, 2]), np.ones([1, 2]), np.array([[1, -1, 0], [1, 0, -1]])]
ls_res_ac = [np.array([1, -1]), np.ones([1, 2]), np.array([[1, -1, 0], [1, 0, -1]])]
ls_res_ab = [subtract_matrix(2), subtract_matrix(2), np.ones([1, 3])]
ls_res_bc = [np.ones([1, 2]), subtract_matrix(2), subtract_matrix(3)]
ls_res_abc = [subtract_matrix(2), subtract_matrix(2), subtract_matrix(3)]

Rsum = reduce(np.kron, ls_sum)
Ra = reduce(np.kron, ls_res_a)
Rb = reduce(np.kron, ls_res_b)
Rc = reduce(np.kron, ls_res_c)
Rac = reduce(np.kron, ls_res_ac)
Rab = reduce(np.kron, ls_res_ab)
Rbc = reduce(np.kron, ls_res_bc)
Rabc = reduce(np.kron, ls_res_abc)

y_sum = np.sum(vec_x) + np.sqrt(2.4) * np.random.normal(size=1)
y_res_a = Ra @ vec_x + np.sqrt(2) * np.random.normal(size=Ra.shape[0])
y_res_b = Rb @ vec_x + np.sqrt(2) * np.random.normal(size=Rb.shape[0])
y_res_c = Rc @ vec_x + np.sqrt(2) * np.random.normal(size=Rc.shape[0])
y_res_ab = Rab @ vec_x + np.sqrt(1) * np.random.normal(size=Rab.shape[0])
y_res_bc = Rbc @ vec_x + np.sqrt(1) * np.random.normal(size=Rbc.shape[0])

y_ac_recon = Mac @ np.linalg.pinv(Rsum) @ y_sum + \
             Mac @ np.linalg.pinv(Ra) @ y_res_a + \
             Mac @ np.linalg.pinv(Rc) @ y_res_c
y_ac = Mac @ vec_x
recon_error = np.linalg.norm(y_ac_recon - y_ac, 1)
print("The L1 error for lsq recon is: ", recon_error)


y_res_ac = cp.Variable(Rac.shape[0])
y_res_abc = cp.Variable(Rabc.shape[0])
objective = cp.Minimize(cp.norm(np.linalg.pinv(Rac) @ y_res_ac, 2) + cp.norm(np.linalg.pinv(Rabc) @ y_res_abc, 2))
constraints = [np.linalg.pinv(Rsum) @ y_sum + \
               np.linalg.pinv(Ra) @ y_res_a + \
               np.linalg.pinv(Rc) @ y_res_c + \
               np.linalg.pinv(Rb) @ y_res_b + \
               np.linalg.pinv(Rbc) @ y_res_bc + \
               np.linalg.pinv(Rab) @ y_res_ab + \
               np.linalg.pinv(Rac) @ y_res_ac + \
               np.linalg.pinv(Rabc) @ y_res_abc >= 0]
problem = cp.Problem(objective, constraints)
problem.solve(solver=cp.ECOS)

y_ac_recon_non_neg = Mac @ np.linalg.pinv(Rsum) @ y_sum + \
               Mac @ np.linalg.pinv(Ra) @ y_res_a + \
               Mac @ np.linalg.pinv(Rc) @ y_res_c + \
               Mac @ np.linalg.pinv(Rac) @ y_res_ac.value
recon_error_non_neg = np.linalg.norm(y_ac_recon_non_neg - y_ac, 1)
print("The L1 error for non neg y-abc recon is: ", recon_error_non_neg)
