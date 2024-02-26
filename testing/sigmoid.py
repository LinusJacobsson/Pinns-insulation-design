from scipy.integrate import solve_bvp
import numpy as np
import matplotlib.pyplot as plt

# Define your sigmoid function
def sigmoid(x, k=10, x0=0.5):
    return 1 / (1 + np.exp(-k * (x - x0)))

# Define the ODE system
def sigmoid_ode(x, y, k, x0, charge):
    return np.vstack((y[1], -charge * sigmoid(x, k=k, x0=x0)))

# Boundary conditions: U(0) = 1000, U(1) = 0
def bc(ya, yb):
    return np.array([ya[0] - 1000, yb[0]])

# Solve the ODE
def solve_ode_for_x(x_values, k=10, x0=0.5, charge=1.0):
    # Initial guess for y and y'
    y_init = np.zeros((2, x_values.size))
    y_init[0] = np.linspace(1000, 0, x_values.size)  # Linear interpolation as initial guess

    # Solve BVP
    sol = solve_bvp(lambda x, y: sigmoid_ode(x, y, k, x0, charge), bc, x_values, y_init)

    return sol

# Create an array of x values
x_values = np.linspace(0, 1, 100)

# Solve the BVP with specified parameters
sol = solve_ode_for_x(x_values, k=10, x0=0.5, charge=1.0)

# Plot the solution U(x)
plt.plot(sol.x, sol.y[0], label='Potential U(x)')
plt.xlabel('x')
plt.ylabel('U(x)')
plt.title('Solution to the ODE with BVP Conditions')
plt.legend()
plt.show()

# Optionally, plot the derivative U'(x) if needed
plt.plot(sol.x, sol.y[1], label="Electric Field U'(x)", color='red')
plt.xlabel('x')
plt.ylabel("U'(x)")
plt.title("Derivative of the Solution U'(x)")
plt.legend()
plt.show()
