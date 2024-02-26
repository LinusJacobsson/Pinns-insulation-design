import jax 
import jax.numpy as jnp
import numpy as np
import optax
import flax.linen as nn
from typing import Sequence, Callable
import matplotlib.pyplot as plt

def analytic(t, x_0, v_0, omega):
    """
    Calculate the analytical solution of the simple harmonic oscillator.

    Args:
        t (array_like): The time points at which the solution is evaluated.
        x_0 (float): The initial displacement of the oscillator.
        v_0 (float): The initial velocity of the oscillator.
        omega (float): The angular frequency of the oscillator.

    Returns:
        array_like: The displacement of the oscillator at each time point.
    """
    A = x_0  # Amplitude from initial displacement
    B = v_0 / omega  # Amplitude from initial velocity
    x = A * np.cos(omega * t) + B * np.sin(omega * t)
    return x


def generate_dataset(N=10, noise_percent=0.0, omega=4, seed=420, x_0=-2, v_0=0):
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

    tmin, tmax = 0.0, 10.0
    t_vals = np.random.uniform(low=tmin, high=tmax, size=(N, 1))
    u_vals = analytic(t=t_vals, omega=omega, x_0=x_0, v_0=v_0)
    noise = np.random.normal(0, u_vals.std(), [N, 1]) * noise_percent
    u_vals += noise
    colloc = jnp.concatenate([t_vals, u_vals], axis=1)

    return colloc, tmin, tmax


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
    

def PINN_f(t, omega, ufunc):
    """
    Physics-informed neural network function representing the dynamics of the system.

    Args:
        t (array_like): Time points.
        omega (float): Angular frequency.
        ufunc (Callable): Function representing the neural network's approximation.

    Returns:
        array_like: Output of the physics-informed function.
    """
    u_t = lambda t: jax.grad(lambda t: jnp.sum(ufunc(t)))(t)
    u_tt = lambda t: jax.grad(lambda t: jnp.sum(u_t(t)))(t)
    return u_tt(t) + omega ** 2 * ufunc(t)  # Second-order differential equation
    

@jax.jit
def uNN(params, t):
    """
    Apply the neural network model to the given inputs.

    Args:
        params (dict): Parameters of the neural network.
        t (array_like): Input data to the network.

    Returns:
        array_like: Output of the neural network.
    """
    u = model.apply(params, t)
    return u


def loss_fun(params, data):
    """
    Calculate the loss function for the neural network training.

    Args:
        params (dict): Parameters of the neural network.
        data (array_like): Training data.

    Returns:
        float: Computed loss value.
    """
    t_c, u_c = data[:, [0]], data[:, [1]]
    ufunc = lambda t: uNN(params, t)

    omega = params["params"]["omega"]  # Retrieve the omega parameter

    mse_u = MSE(u_c, ufunc(t_c)) # data loss
    mse_f = jnp.mean(PINN_f(t_c, omega, ufunc) ** 2) # diff eq. loss

    return mse_f + 1000 * mse_u

@jax.jit
def update(opt_state, params, data):
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
    grads = jax.jit(jax.grad(loss_fun, 0))(params, data)

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



features = [8, 8, 8, 1] # size of network
omega = 4 # true value of omega
x_0 = -2 # initial position
v_0 = 0 # initial velocity
N = 100 # number of sampled points
data, tmin, tmax = generate_dataset(N=N, omega=omega, x_0=x_0, v_0=v_0)
model, params, optimizer, opt_state = init_process(features)


epochs = 20_000
for epoch in range(epochs):
    opt_state, params = update(opt_state,params,data)
    current_omega = params["params"]["omega"][0]
    # print loss and epoch info
    if epoch%(1000) ==0:
        print(f'Epoch = {epoch},\tloss = {loss_fun(params,data):.3e},\tomega = {current_omega:.3f}')


omega_calc = params["params"]["omega"][0]
print(f"The real value of the parameter is omega = {omega}")
print(f"The calculated value for the parameter is k_calc = {omega_calc:.7f}.")
print(f"This corresponds to a {100*(omega_calc-omega)/omega:.5f}% error.")

# Generate a set of time points for evaluation
t_eval = np.linspace(tmin, tmax, 500)[:, None]

# Compute the analytical solution
solution = analytic(t=t_eval, x_0=-2, v_0=0, omega = omega)

# Compute the neural network prediction
nn_solution = uNN(params, jnp.array(t_eval))

# Plotting
plt.figure(figsize=(10, 6))
plt.plot(t_eval, solution, label='Analytical Solution', color='blue')
plt.scatter(data[:, 0], analytic(data[:, 0], -2, 0, omega), color = 'red')
plt.plot(t_eval, nn_solution, label='NN Prediction', linestyle='--', color='red')
plt.xlabel('Time')
plt.ylabel('Displacement')
plt.title('Comparison of Analytical and NN Solutions for the SHO')
plt.legend()
plt.show()
