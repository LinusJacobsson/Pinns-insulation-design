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


def generate_dataset(N=100, noise_percent=0.0, seed=420, charge = 100):
   
    np.random.seed(seed)
    xmin, xmax = 0.0, 1.0
    x_vals = np.random.uniform(low=xmin, high=xmax, size=(N, 1))
    return x_vals

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
                x = nn.tanh(x)
        return x
    
@jax.jit
def MSE(true, pred):
    return jnp.mean((true - pred) ** 2)

def PINN_f(x, ufunc, params):
    epsilon = 2*8.85e-12
    q = 1.6e-19
    n0 = 1e16
    L_c = 0.01
    U_c = 10
   
    u_x = lambda x: jax.grad(lambda x: jnp.sum(ufunc(x)))(x)
    u_xx = lambda x: jax.grad(lambda x: jnp.sum(u_x(x)))(x)
    normalized_constant = (1e-6*q)/(1e3*epsilon)
    return u_xx(x)*U_c/(L_c**2) + ((L_c**3)*q*n0*(x-0.5)**3)/epsilon
    
@jax.jit
def uNN(params, x):
    u = model.apply(params, x)
    return u


@jax.jit
def loss_fun(params, data_fitting, data_de, U_0, U_1):
    # Split data fitting for data loss calculation
    x_data, e_data = data_fitting[:, [0]], data_fitting[:, [1]]
    ufunc = lambda x: uNN(params, x).squeeze()  # Ensure this is scalar for each x

    # Compute DE loss (function_loss or mse_f) using x_de
    mse_f = MSE(PINN_f(data_de, ufunc, params), 0)

    # Compute the first boundary condition loss for U at x = xmin
    bc_loss1 = MSE(ufunc(jnp.array([[0.0]])), U_0)
    bc_loss2 = MSE(ufunc(jnp.array([[1.0]])), U_1)    

    # Combine losses
    total_loss =  1e0*mse_f + 10000*bc_loss1 + 10000*bc_loss2
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
    lr = optax.piecewise_constant_schedule(1e-2, {40_000: 5e-3, 500_000: 1e-3})
    optimizer = optax.adam(lr)
    opt_state = optimizer.init(params)

    return model, params, optimizer, opt_state


features = [8, 8, 1] # size of network

N_data = 100 # number of sampled points
N_equation = 100

CHARGE_GUESS = 5.0*10**4
# Normalized U-values
U_0 = 1
U_1 = 0

data = generate_dataset(N=1000)
#rint(data_fitting[:, 1])

print(f"Starting training")
model, params, optimizer, opt_state = init_process(features, CHARGE_GUESS)
epochs = 500_000
for epoch in range(epochs):
    opt_state, params = update(opt_state, params, data, data, U_0, U_1)
    
    if epoch % 1000 == 0:
        
        ufunc = lambda x: uNN(params, x).squeeze()  # Ensure this is scalar for each x

        # Compute DE loss using DE data
        mse_f = jnp.mean(PINN_f(data, ufunc, params) ** 2)

        # Compute boundary condition losses
        bc_loss1 = MSE(ufunc(jnp.array([[0.0]])), U_0)
        bc_loss2 = MSE(ufunc(jnp.array([[1.0]])), U_1)

        total_loss = mse_f + bc_loss1 + bc_loss2
        current_charge = params["params"]["charge"][0]

        # Print the detailed losses
        print(f'Epoch = {epoch}, Total Loss = {total_loss:.3e}, DE Loss = {mse_f:.3e}, BC Loss 1 = {bc_loss1:.3e}, BC Loss 2 = {bc_loss2:.3e}')

print(f"Training complete")
current_charge = params["params"]["charge"][0]

# Assuming x_eval is a numpy array of evaluation points
x_eval = np.linspace(0, 1, 1000)[:, None]
# Vectorize the electric_field_single function to work over batches of inputs
electric_field_batch = jax.jit(jax.vmap(electric_field_single, in_axes=(None, 0)))
# Compute predicted potential and electric field using your neural network

nn_solution = uNN(params, jnp.array(x_eval)).reshape(-1)
e_field_nn = electric_field_batch(params, jnp.array(x_eval)).reshape(-1)
#field_nn = e_field_nn.at[-1].set(e_field_nn[-2]) # Fixes last value
fig, axs = plt.subplots(1, 2, figsize=(15, 5))


field_path = '/Users/linus/Desktop/Github/Pinns-insulation-design/poisson/data/Case3_field.csv'
potential_path = '/Users/linus/Desktop/Github/Pinns-insulation-design/poisson/data/Case3_Potential.csv'
field_df = pd.read_csv(field_path, skiprows=7)
x_data= field_df[['x-coordinate (m)']].values
e_data = field_df[['Electric field norm']].values
potential_df = pd.read_csv(potential_path, skiprows=7)
potential_data = potential_df[['Electric potential']].values
# Plot true vs predicted potential
axs[0].plot(x_data, potential_data, label='True Potential U(x)', linestyle='-', color='blue')
axs[0].plot(x_eval/100, 1e1*nn_solution, label='Predicted Potential U(x)', linestyle='--', color='red')
axs[0].set_xlabel('x (m)')
axs[0].set_ylabel('U(x) (V)')
axs[0].legend()
axs[0].set_title('Potential U(x)')



# Plot true vs predicted electric field
# axs[1].scatter(data_fitting[:, 0], data_fitting[:, 1], color='k', label='Training Data')  # Uncomment if you want to show training data points
axs[1].plot(x_data, e_data, label='True Electric Field E(x)', linestyle='-', color='blue')
axs[1].plot(x_eval/100, 1e3*e_field_nn, label="Predicted Electric Field E(x)", linestyle='--', color='red')

axs[1].set_xlabel('x (m)')
axs[1].set_ylabel("E(x) (V/m)")
axs[1].legend()
axs[1].set_title("Electric Field E(x)")
#axs[1].set_ylim([99000, 101000])  # Set y-axis range

plt.tight_layout()
plt.show()
