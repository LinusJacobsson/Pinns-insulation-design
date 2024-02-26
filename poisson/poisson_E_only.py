import jax 
import jax.numpy as jnp
import numpy as np
import optax
import flax.linen as nn
from typing import Sequence, Callable
import matplotlib.pyplot as plt


@jax.jit
def electric_field(x, const):
    return (const*x**2)/2 +1



def generate_dataset(N=10, noise_percent=0.0, seed=420, charge = 1000):
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
    xmin, xmax = 0.0, 1.0
    x_vals = np.random.uniform(low=xmin, high=xmax, size=(N, 1))
    u_vals = electric_field(x=x_vals, const=charge)
    noise = np.random.normal(0, u_vals.std(), [N, 1]) * noise_percent
    u_vals += noise
    colloc = jnp.concatenate([x_vals, u_vals], axis=1)

    return colloc, xmin, xmax


class MLP(nn.Module):
    features: Sequence[int]
    charge_value: float  # Add a class attribute for the charge value

    def setup(self):
        # Define a custom initializer function for the charge parameter
        def charge_init(key, shape):
            return jnp.full(shape, self.charge_value)  # Initialize charge with the specified value

        # Initialize the charge parameter with the custom initializer
        self.charge = self.param("charge", charge_init, (1,))
        self.layers = [nn.Dense(features=feat) for feat in self.features]

    def __call__(self, inputs):
        x = inputs
        for layer in self.layers:
            x = layer(x)
            x = nn.relu(x)  # Example activation function
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
    

def PINN_f(x, ufunc, params):
    charge = params["params"]["charge"][0]
    u_x = lambda x: jax.grad(lambda x: jnp.sum(ufunc(x)))(x)
    u_xx = lambda x: jax.grad(lambda x: jnp.sum(u_x(x)))(x)
    return u_x(x) - charge*x   

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

@jax.jit
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
    x_c, e_vals = data[:, [0]], data[:, [1]]
    ufunc = lambda x: uNN(params, x)

    # Compute the gradient of the neural network output w.r.t. its input
    du_dx = lambda x: jax.grad(lambda x: ufunc(x)[0, 0])(x)
    # Differential equation loss
    mse_f = jnp.mean(PINN_f(x_c, ufunc, params) ** 2)
    mse_data = MSE(ufunc(x_c), e_vals)
    # Boundary condition losses
    bc_loss1 = jnp.mean((ufunc(jnp.array([[xmin]])) - 1) ** 2)
    #bc_loss2 = jnp.mean((ufunc(jnp.array([[xmax]])) - 1.5) ** 2)

    # Total loss
    total_loss = mse_f + bc_loss1 + 100*mse_data

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


def init_process(feats, charge_guess):
    """
    Initialize the neural network model, its parameters, the optimizer, and the optimizer state.

    Args:
        feats (Sequence[int]): The number of neurons in each layer of the MLP.

    Returns:
        tuple: Initialized model, parameters, optimizer, and optimizer state.
    """
    model = MLP(features=feats, charge_value=charge_guess)

    key1, key2 = jax.random.split(jax.random.PRNGKey(420), num=2)

    dummy_in = jax.random.normal(key1, (1,))
    params = model.init(key2, dummy_in)

    lr = optax.piecewise_constant_schedule(1e-2, {200_000: 5e-3, 250_000: 1e-3})
    optimizer = optax.adam(lr)
    opt_state = optimizer.init(params)

    return model, params, optimizer, opt_state



features = [16, 16, 1] # size of network

N = 500 # number of sampled points

CHARGE = 100.0
CHARGE_GUESS = 50.0

U_0 = 1
U_1 = 0
data, xmin, xmax = generate_dataset(N=N, charge=CHARGE)
model, params, optimizer, opt_state = init_process(features, CHARGE_GUESS)
epochs = 20_000
for epoch in range(epochs):
    opt_state, params = update(opt_state,params,data, U_0, U_1)
    current_charge = params["params"]["charge"][0]
    # print loss and epoch info
    if epoch%(1000) ==0:
        print(f'Epoch = {epoch},\tloss = {loss_fun(params, data, xmin, xmax, U_0, U_1):.3e},\tcharge = {current_charge:.3f}')

# Generate a set of time points for evaluation
x_eval = np.linspace(xmin, xmax, 500)[:, None]

# Compute the neural network prediction
nn_solution = uNN(params, jnp.array(x_eval))

# Evaluate the analytical electric field over the same range
analytical_e_field = electric_field(x_eval, const=CHARGE)

# Plotting
plt.figure(figsize=(8, 6))
plt.plot(x_eval, analytical_e_field, label='Analytical Electric Field', color='blue')
plt.plot(x_eval, nn_solution, label='NN Predicted Electric Field', linestyle='--', color='red')
plt.xlabel('x')
plt.ylabel('E(x)')
plt.title(f'True charge: {CHARGE:.2f}, Predicted charge: {current_charge:.2f}, Error: {100*(CHARGE-current_charge)/CHARGE:.2f}%')
plt.legend()
plt.grid()

plt.tight_layout()
plt.show()