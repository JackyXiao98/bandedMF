import scipy.optimize as optimize
import numpy as np

# Define the objective function
def objective(x):
    return 1/x[0] + 1/x[1] + 1/x[2] + 1/x[3]

# Define the constraints
constraints = [
    {'type': 'ineq', 'fun': lambda x: x[0]},
    {'type': 'ineq', 'fun': lambda x: 0.4 - x[0]},
    {'type': 'ineq', 'fun': lambda x: x[1]},
    {'type': 'ineq', 'fun': lambda x: 1 - x[0] - x[1]},
    {'type': 'ineq', 'fun': lambda x: x[2]},
    {'type': 'ineq', 'fun': lambda x: 2.8 - x[0] - x[1] - x[2]},
    {'type': 'ineq', 'fun': lambda x: x[3]},
    {'type': 'ineq', 'fun': lambda x: 4 - x[0] - x[1] - x[2] - x[3]}
]

# Initial guess for the variables
x0 = np.array([0.25, 0.25, 0.25, 0.25])

# Solve the optimization problem
solution = optimize.minimize(objective, x0, constraints=constraints)

# Output the results
print("Optimal value of the objective function:", solution.fun)
print("Optimal values of the variables (x1, x2, x3, x4):", solution.x)
