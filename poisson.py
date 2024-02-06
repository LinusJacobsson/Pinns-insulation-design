import jax 
import jax.numpy as jnp
import numpy as np
import optax
import flax.linen as nn
from typing import Sequence, Callable
import matplotlib.pyplot as plt


# Hard coded for the boundary conditions in the default arguments
# This is n(x) = x^2
"""def analytic(x, U_0 = 1, U_1 = 0, epsilon = 8.854e-12):
   
    q = 1.602e-19 # elementary charge
    
    A = -(q/(12*epsilon))
    B = (q/(12*epsilon)-1)
    C = 1
    return A * x**4 + B * x + C"""

# assuming y(1) = y'(1) = 1
def analytic(x):
    return -(x**3)/6 + 119*x/30 + 1



@jax.jit
def electric_field_single(params, x):
    """
    Compute the electric field as the negative gradient of the potential U for a single input x.

    Args:
        params (dict): Parameters of the neural network.
        x (array_like): Single input data to the network.

    Returns:
        array_like: Electric field computed as -dU/dx for the single input x.
    """
    # Compute the gradient of the neural network output with respect to its input
    dU_dx = jax.grad(lambda x: jnp.squeeze(uNN(params, x)))(x)
    
    # Return the negative of the gradient to represent the electric field
    return -dU_dx


def generate_dataset(N=100, noise_percent=0.0, seed=420, U_0=1, U_1=1):
    """
    Generate a dataset for training the neural network.

    Args:
        N (int, optional): Number of data points. Default is 10.
        noise_percent (float, optional): Percentage of noise to add to the data. Default is 0.0.
        omega (float, optional): Angular frequency for the simple harmonic oscillator. Default is 4.
        seed (int, optional): Random seed for reproducibility. Default is 420.
        x_0 (float, optional): Initial displacement. Default is -2.
        v_0 (float, optional): Initial velocity. Default is 0.

    Returns:
        tuple: A tuple containing the dataset and the min/max values of the time domain.
    """
    np.random.seed(seed)
    xmin, xmax = 0.0, 5.0
    x_vals = np.random.uniform(low=xmin, high=xmax, size=(N, 1))
    u_vals = analytic(x=x_vals)
    noise = np.random.normal(0, u_vals.std(), [N, 1]) * noise_percent
    u_vals += noise
    colloc = jnp.concatenate([x_vals, u_vals], axis=1)

    return colloc, xmin, xmax


class MLP(nn.Module):
    """
    A simple Multi-Layer Perceptron (MLP) model using the Flax Linen API.

    Attributes:
        features (Sequence[int]): The number of neurons in each layer of the MLP.
        omega_init (Callable, optional): Initializer for the omega parameter.
    """

    features: Sequence[int]
    # We also add an initializer for the omega parameter
    omega_init: Callable = jax.nn.initializers.ones

    def setup(self):
        # include the omega parameter during setup
        omega = self.param("omega", self.omega_init, (1,))
        self.layers = [nn.Dense(features=feat, use_bias=True) for feat in self.features]
        
    def __call__(self, inputs):
        x = inputs
        for idx, layer in enumerate(self.layers):
            x = layer(x)
            if idx != len(self.layers)-1:
                x = jnp.tanh(x)
        return x

@jax.jit
def MSE(true, pred):
    """
    Calculate the mean squared error between the true values and predictions.

    Args:
        true (array_like): True values.
        pred (array_like): Predicted values.

    Returns:
        float: The mean squared error.
    """
    return jnp.mean((true - pred) ** 2)
    

def PINN_f(x, ufunc):
    q = 1.602e-19 # elementary charge
    epsilon = 8.854e-12

    u_x = lambda x: jax.grad(lambda x: jnp.sum(ufunc(x)))(x)
    u_xx = lambda x: jax.grad(lambda x: jnp.sum(u_x(x)))(x)
    return u_xx(x) + x # Laplace equation for n(x) = log(x)   

@jax.jit
def uNN(params, x):
    """
    Apply the neural network model to the given inputs.

    Args:
        params (dict): Parameters of the neural network.
        t (array_like): Input data to the network.

    Returns:
        array_like: Output of the neural network.
    """
    u = model.apply(params, x)
    return u


def loss_fun(params, data, xmin, xmax, U_0, U_1):
    """
    Calculate the loss function for the neural network training including new boundary conditions u(1) = 1 and u'(1) = 1.

    Args:
        params (dict): Parameters of the neural network.
        data (array_like): Collocation points for differential equation.
        xmin (float): Minimum boundary point.
        xmax (float): Maximum boundary point (assumed to be 1 for the boundary condition u(1) = 1 and u'(1) = 1).
        U_0 (float): Solution value at xmin.
        U_1 (float): Solution value at xmax, which is 1 according to the boundary condition u(1) = 1.

    Returns:
        float: Computed loss value.
    """
    x_c, _ = data[:, [0]], data[:, [1]]
    ufunc = lambda x: uNN(params, x)

    # Compute the gradient of the neural network output w.r.t. its input
    du_dx = lambda x: jax.grad(lambda x: ufunc(x)[0, 0])(x)
    # Differential equation loss
    mse_f = jnp.mean(PINN_f(x_c, ufunc) ** 2)

    # Boundary condition losses
    bc_loss1 = jnp.mean((ufunc(jnp.array([[xmin]])) - 1) ** 2)

    # Derivative boundary condition at x = 1
    bc_loss2 = jnp.mean((ufunc(jnp.array([[xmax]]))) ** 2)

    # Total loss
    total_loss = mse_f + bc_loss1 + bc_loss2

    return total_loss


@jax.jit
def update(opt_state, params, data, U_0, U_1):
    """
    Update the parameters of the model using the optimizer.

    Args:
        opt_state (any): The state of the optimizer.
        params (dict): Parameters of the neural network.
        data (array_like): Training data.

    Returns:
        tuple: Updated optimizer state and neural network parameters.
    """
    # Get the gradient w.r.t to MLP params
    grads = jax.jit(jax.grad(loss_fun, 0))(params, data, xmin, xmax, U_0, U_1)

    # Update params
    updates, opt_state = optimizer.update(grads, opt_state)
    params = optax.apply_updates(params, updates)

    return opt_state, params


def init_process(feats):
    """
    Initialize the neural network model, its parameters, the optimizer, and the optimizer state.

    Args:
        feats (Sequence[int]): The number of neurons in each layer of the MLP.

    Returns:
        tuple: Initialized model, parameters, optimizer, and optimizer state.
    """
    model = MLP(features=feats)

    key1, key2 = jax.random.split(jax.random.PRNGKey(420), num=2)

    dummy_in = jax.random.normal(key1, (1,))
    params = model.init(key2, dummy_in)

    lr = optax.piecewise_constant_schedule(1e-2, {15_000: 5e-3, 80_000: 1e-3})
    optimizer = optax.adam(lr)
    opt_state = optimizer.init(params)

    return model, params, optimizer, opt_state



features = [8, 8, 1] # size of network

N = 20 # number of sampled points
data, xmin, xmax = generate_dataset(N=N)
model, params, optimizer, opt_state = init_process(features)

U_0 = 1
U_1 = 0
epochs = 50_000
for epoch in range(epochs):
    opt_state, params = update(opt_state,params,data, U_0, U_1)
    current_omega = params["params"]["omega"][0]
    # print loss and epoch info
    if epoch%(1000) ==0:
        print(f'Epoch = {epoch},\tloss = {loss_fun(params, data, xmin, xmax, U_0, U_1):.3e},\tomega = {current_omega:.3f}')

# Generate a set of time points for evaluation
x_eval = np.linspace(xmin, xmax, 500)[:, None]

# Compute the analytical solution
solution = analytic(x=x_eval)

# Compute the neural network prediction
nn_solution = uNN(params, jnp.array(x_eval))

# Vectorize the electric_field_single function to work over batches of inputs
electric_field_batch = jax.jit(jax.vmap(electric_field_single, in_axes=(None, 0)))

# Now use electric_field_batch to compute the electric field for all points in x_eval
e_field_nn = electric_field_batch(params, jnp.array(x_eval).reshape(-1, 1))

# Plotting both the solution and the electric field
plt.figure(figsize=(12, 8))

# Plot the potential
plt.subplot(2, 1, 1)
plt.plot(x_eval, solution, label='Analytical Solution', color='blue')
plt.scatter(data[:, 0], analytic(data[:, 0]), color='red', label='Training Data')
plt.plot(x_eval, nn_solution, label='NN Prediction', linestyle='--', color='red')
plt.xlabel('x')
plt.ylabel('U(x)')
plt.title('Potential U(x)')
plt.legend()
plt.grid()

# Plot the electric field
plt.subplot(2, 1, 2)
plt.plot(x_eval, e_field_nn, label='NN Predicted Electric Field', linestyle='--', color='green')
plt.xlabel('x')
plt.ylabel('E(x)')
plt.title('Electric Field E(x)')
plt.legend()
plt.grid()

plt.tight_layout()
plt.show()