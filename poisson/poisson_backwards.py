import jax 
import jax.numpy as jnp
import numpy as np
import optax
import flax.linen as nn
from typing import Sequence, Callable
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
import pandas as pd

@jax.jit
def electric_field_single(params, x):
    # Compute the gradient of the neural network output with respect to its input
    dU_dx = jax.grad(lambda x: jnp.squeeze(uNN(params, x)))(x)
    
    # Return the negative of the gradient to represent the electric field
    return -dU_dx

# Your sigmoid function
def charge_distribution(x, d = 0.01):
    return (x - 1)**3


def generate_dataset(filename, noise_percent=0.0):
    df = pd.read_csv(filename, skiprows=8, header=None, names=['x-coordinate (m)', 'Electric field norm'])
    dataset = df.values
    
    # Normalize x values to the range [0, 1]
    x_min, x_max = np.min(dataset[:, 0]), np.max(dataset[:, 0])
    dataset[:, 0] = (dataset[:, 0] - x_min) / (x_max - x_min)
    
    # Normalize Electric field values to the range [0, 1]
    E_min, E_max = np.min(dataset[:, 1]), np.max(dataset[:, 1])
    print(f"E_min: {E_min}")
    print(f"E_max: {E_max}")
    dataset[:, 1] = (dataset[:, 1] - E_min) / (E_max - E_min)
    
    # Add noise if needed
    if noise_percent > 0:
        # Adjust the noise scale to the normalized data
        noise_scale = noise_percent * (E_max - E_min)
        noise = np.random.normal(0, noise_scale, dataset.shape[0])
        dataset[:, 1] += noise
    
    return dataset



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
        for idx, layer in enumerate(self.layers):
            x = layer(x)
            if idx != len(self.layers)-1:
                x = nn.sigmoid(x)
        return x
    
@jax.jit
def MSE(true, pred):
    return jnp.mean((true - pred) ** 2)
    

def PINN_f(x, ufunc, params):
    epsilon = 2*8.85e-12
    q = 1.6e-19
    n0 = params["params"]["charge"][0]
    scaling_factor = 1e16
    scaled_n0 = n0*scaling_factor
    x_unscaled = x * 100
    u_x = lambda x: jax.grad(lambda x: jnp.sum(ufunc(x)))(x)
    u_xx = lambda x: jax.grad(lambda x: jnp.sum(u_x(x)))(x)

    return 1e8*(u_xx(x)*(epsilon) + q*scaled_n0*charge_distribution(x))

@jax.jit
def uNN(params, x):
    u = model.apply(params, x)
    return u


@jax.jit
def loss_fun(params, data_fitting, data_de, U_0, U_1):
    # Split data fitting for data loss calculation
    x_data, e_data = data_fitting[:, [0]], data_fitting[:, [1]]
    ufunc = lambda x: uNN(params, x).squeeze()  # Ensure this is scalar for each x

    # Compute data loss as before
    #du_dx = jax.vmap(lambda x: -jax.grad(ufunc)(x))(x_data)
    du_dx = lambda x: -jax.grad(lambda x: jnp.sum(ufunc(x)))(x)

    data_loss = MSE(du_dx(x_data), e_data)

    # Split DE data for function loss calculation
    x_de, _ = data_de[:, [0]], data_de[:, [1]]  # Assuming DE data has the same format

    # Compute DE loss (function_loss or mse_f) using x_de
    mse_f = MSE(PINN_f(x_de, ufunc, params), 0)

    # Compute the first boundary condition loss for U at x = xmin
    bc_loss1 = MSE(ufunc(jnp.array([[0.0]])), U_0)
    bc_loss2 = MSE(ufunc(jnp.array([[1.0]])), U_1)    

    # Combine losses
    total_loss =  20*bc_loss1 + 20*bc_loss2 + 1000*data_loss + mse_f
    return total_loss



@jax.jit
def update(opt_state, params, data_fitting, data_equation, U_0, U_1):
    # Get the gradient w.r.t to MLP params
    grads = jax.jit(jax.grad(loss_fun, 0))(params, data_fitting, data_equation, U_0, U_1)

    # Update params
    updates, opt_state = optimizer.update(grads, opt_state)
    params = optax.apply_updates(params, updates)

    return opt_state, params


def init_process(feats, charge_guess):
    model = MLP(features=feats, charge_value=charge_guess)
    key1, key2 = jax.random.split(jax.random.PRNGKey(420), num=2)
    dummy_in = jax.random.normal(key1, (1,))
    params = model.init(key2, dummy_in)
    lr = optax.piecewise_constant_schedule(1e-2, {50_000: 5e-3, 85_000: 1e-3})
    optimizer = optax.adam(lr)
    opt_state = optimizer.init(params)

    return model, params, optimizer, opt_state



features = [128, 128, 1] # size of network

N_data = 100 # number of sampled points
N_equation = 1000 

CHARGE_GUESS = 5.0

U_0 = 1/100
U_1 = 0
filename = 'poisson/data/Case3_Field.csv'
potential_data = 'poisson/data/Case3_Potential.csv'
data_fitting = generate_dataset(filename)
data_equation = generate_dataset(filename)  # Assuming generate_dataset can accept noise_percent=0.0 to generate without noise
potential_values = generate_dataset(potential_data)
print(potential_values)
x_data_normalized = data_fitting[:, 0]
e_data_normalized = data_fitting[:, 1]

"""plt.plot(x_data_normalized, e_data_normalized, label='Normalized data')
#plt.ylim([-1001, -999])
plt.xlabel('x')
plt.ylabel('e-field')
plt.title('Predicted Derivative of the Solution')
#plt.ylim([-995, -1005])
plt.legend()
plt.show()"""
print(f"Starting training")
model, params, optimizer, opt_state = init_process(features, CHARGE_GUESS)
epochs = 10_000
for epoch in range(epochs):
    opt_state, params = update(opt_state, params, data_fitting, data_equation, U_0, U_1)
    
    if epoch % 1000 == 0:
        # Split datasets for logging
        x_data_fitting, u_data_fitting = data_fitting[:, [0]], data_fitting[:, [1]]
        x_de = data_equation[:, [0]]  # Assuming DE data format matches

        ufunc = lambda x: uNN(params, x).squeeze()  # Ensure this is scalar for each x

        # Compute gradients and losses for data fitting
        du_dx = jax.vmap(lambda x: -jax.grad(ufunc)(x))
        data_loss = MSE(du_dx(x_data_fitting), u_data_fitting)

        # Compute DE loss using DE data
        mse_f = jnp.mean(PINN_f(x_de, ufunc, params) ** 2)

        # Compute boundary condition losses
        bc_loss1 = MSE(ufunc(jnp.array([[0.0]])), U_0)
        bc_loss2 = MSE(ufunc(jnp.array([[1.0]])), U_1)

        total_loss = 1e-3*mse_f + 1000*bc_loss1 + 1000*bc_loss2 + 100*data_loss
        current_charge = params["params"]["charge"][0]

        # Print the detailed losses
        print(f'Epoch = {epoch}, Total Loss = {total_loss:.3e}, DE Loss = {mse_f:.3e}, BC Loss 1 = {bc_loss1:.3e}, BC Loss 2 = {bc_loss2:.3e}, Data Loss = {data_loss:.3e},Charg={current_charge:.3e}')

print(f"Training complete")
current_charge = params["params"]["charge"][0]

# Assuming x_eval is a numpy array of evaluation points
x_eval = np.linspace(0, 1, 100)[:, None]
# Vectorize the electric_field_single function to work over batches of inputs
electric_field_batch = jax.jit(jax.vmap(electric_field_single, in_axes=(None, 0)))
# Compute predicted potential and electric field using your neural network

nn_solution = uNN(params, jnp.array(x_eval)).reshape(-1)
e_field_nn = electric_field_batch(params, jnp.array(x_eval)).reshape(-1)
e_field_nn = e_field_nn.at[-1].set(e_field_nn[-2]) # Fixes last value
fig, axs = plt.subplots(1, 3, figsize=(15, 5))

# Plot true vs predicted potential
axs[0].plot(potential_values[:, 0], potential_values[:, 1], label='True Potential U(x)', color='blue')
axs[0].plot(x_eval, nn_solution, label='Predicted Potential U(x)', linestyle='--', color='red')
axs[0].set_xlabel('x (m)')
axs[0].set_ylabel('U(x) (V)')
axs[0].legend()
axs[0].set_title('Potential U(x)')

# Plot true vs predicted electric field
axs[1].plot(data_fitting[:,0], data_fitting[:, 1], label='True Electric Field U\'(x)', color='blue')
# axs[1].scatter(data_fitting[:, 0], data_fitting[:, 1], color='k', label='Training Data')  # Uncomment if you want to show training data points
axs[1].plot(x_eval, e_field_nn, label="Predicted Electric Field E(x)", linestyle='--', color='red')
axs[1].set_xlabel('x (m)')
axs[1].set_ylabel("E(x) (V/m)")
axs[1].legend()
axs[1].set_title("Electric Field E(x)")
#axs[1].set_ylim([990, 1010])  # Set y-axis range


# Plot the sigmoid function n(x)
axs[2].plot(data_fitting[:, 0], 0*charge_distribution(data_fitting[:, 0]) , label='Initial charge distribution)', color='green')
axs[2].plot(data_fitting[:, 0], current_charge*charge_distribution(data_fitting[:, 0]), linestyle='--', label='Predicted Sigmoid', color='red')
axs[2].set_xlabel('x (m)')
axs[2].set_ylabel('n(x) (m^-3)')
axs[2].legend()
axs[2].set_title(f'Predicted n0: {current_charge:.2e}, True n0 = 0')
axs[2].set_ylim([-1e-3, 1e-3])  # Set y-axis range

plt.tight_layout()
plt.show()
