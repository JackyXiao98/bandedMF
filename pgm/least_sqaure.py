import numpy as np
import cvxpy as cp
from functools import reduce


np.random.seed(23)
np.set_printoptions(precision=3)
vec_x = np.array([0, 0, 0, 0, 2, 0, 0, 0, 2, 0, 0, 1])
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
print("The L1 error for Mac is: ", error)
