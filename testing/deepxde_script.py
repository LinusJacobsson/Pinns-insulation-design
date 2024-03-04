import deepxde as dde
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import deepxde as dde
import numpy as np
import tensorflow as tf
import pandas as pd

def ddy(x, y):
    return dde.grad.hessian(y, x, component=0)

def dddy(x, y):
    return dde.grad.jacobian(ddy(x, y), x)

def pde1(x, y):
    epsilon_0 = 8.85e12
    epsilon_r = 2
    q = 1.60e-19
    dy_xx = ddy(x, y)
    return [epsilon_0*epsilon_r*dy_xx + q*n0*x]



def pde(x, y):
    epsilon_0 = 8.85e12
    epsilon_r = 2
    q = 1.60e-19
    dy_xx = ddy(x, y)
    return [dy_xx]

def boundary_l(x, on_boundary):
    return on_boundary and np.isclose(x[0], 0)

def boundary_r(x, on_boundary):
    return on_boundary and np.isclose(x[0], size)

# Define a trainable constant
n0 = dde.Variable(1.0)

size = 0.01
geom = dde.geometry.Interval(0, size)
file_path = '/Users/linus/Desktop/Github/Pinns-insulation-design/poisson/data/Case2_field.csv'
data_df = pd.read_csv(file_path, skiprows=7)
x_data= data_df[['x-coordinate (m)']].values
# Normalize x_data to the range [0, 1]
x_min = np.min(x_data)
x_max = np.max(x_data)
x_data_normalized = (x_data - x_min) / (x_max - x_min)
e_data = -data_df[['Electric field norm']].values
e_min = np.min(e_data)
e_max = np.max(e_data)
e_data_normalized = (e_data - e_min) / (e_max - e_min)
# Number of points
num_points = 300
# Generate x-values from 0 to 1
#x_data = np.linspace(0, size, num_points).reshape(-1, 1)
# Generate constant y-values of 10
#y_data = np.full((num_points, 100), -1000)

# Adjust boundary conditions to match your problem
bc_left = dde.icbc.DirichletBC(geom, lambda x: 10, boundary_l, component=0)
bc_right = dde.icbc.DirichletBC(geom, lambda x: 0, boundary_r, component=0)
bc_data = dde.icbc.PointSetBC(x_data, e_data, component=1)


data = dde.data.PDE(
    geom,
    pde,
    [bc_left, bc_right, bc_data],
    num_domain=100,
    num_boundary=2,
    #solution=func,
    num_test=1,
)
layer_size = [1] + [128] * 2 + [1]
activation = "relu"
initializer = "Glorot uniform"
net = dde.nn.FNN(layer_size, activation, initializer)


def modify_output(X, Y):
    dydx = tf.gradients(Y, X)[0]
    final_output = tf.concat([Y, dydx], axis=1)
    return final_output


net.apply_output_transform(modify_output)

model = dde.Model(data, net)
loss_weights = [1, 1, 1, 1]  # Adjust these values based on your specific needs
model.compile("adam", lr=0.001, loss_weights=loss_weights)

losshistory, train_state = model.train(iterations=20000)
#dde.saveplot(losshistory, train_state, issave=True, isplot=True)

dde.optimizers.set_LBFGS_options(maxiter=1000)
model.compile("L-BFGS")
model.train()

x_eval = np.linspace(0, size, 300)[:, None]  # Generate 1000 points in the interval [0, 1]
y_pred = model.predict(x_eval)
# Assuming y_pred is structured as [Y, dY/dX]
y_solution = y_pred[:, 0]  # First column: solution values
dy_dx_solution = y_pred[:, 1]  # Second column: derivative values

# Plot the predicted solution (Y)
plt.figure(figsize=(10, 4))
plt.subplot(1, 2, 1)  # 1 row, 2 columns, first plot
plt.plot(x_eval, y_solution, label='Predicted Solution $Y$')
plt.xlabel('x')
plt.ylabel('Y(x)')
plt.title('Predicted Solution of the Differential Equation')
plt.legend()

# Plot the predicted derivative of the solution (dY/dX)
plt.subplot(1, 2, 2)  # 1 row, 2 columns, second plot
plt.plot(x_eval, dy_dx_solution, label='Predicted Derivative $dY/dX$')
#plt.ylim([-1001, -999])
plt.xlabel('x')
plt.ylabel('dY/dX')
plt.title('Predicted Derivative of the Solution')
plt.legend()

plt.tight_layout()  # Adjust layout to not overlap
plt.show()

n0_value = model.sess.run(n0.value())
print("Value of the trainable constant n0:", n0_value)