import numpy as np
from scipy.optimize import linprog

# Coefficients of the objective function (to be minimized)
c = np.array([2, 3])  # Objective: 2x1 + 3x2

# Equality constraint (A_eq x = b_eq)
A_eq = np.array([[1, 1]])  # x1 + x2 = 3
b_eq = np.array([3])

# Inequality constraint (A_ub x <= b_ub)
A_ub = np.array([[-1, 1]])  # -x1 + x2 ≤ -1 (equivalent to x1 - x2 ≥ 1)
b_ub = np.array([-1])

# Bounds (x >= 0)
bounds = [(0, None), (0, None)]  # x1, x2 ≥ 0

# Solve the LP problem
result = linprog(c, A_eq=A_eq, b_eq=b_eq, A_ub=A_ub, b_ub=b_ub, bounds=bounds, method='highs')

# Extract and display the solution
print("Optimal value (minimized cost):", result.fun)
print("Optimal solution (x1, x2):", result.x)