import jax 
import jax.numpy as jnp
import numpy as np
import optax
import flax.linen as nn
from typing import Sequence, Callable
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
import pandas as pd
from jax import lax
from dataclasses import dataclass, field, asdict

import logging
import pickle

def create_schedule_description(initial_lr, milestones):
    description = [f"Initial LR: {initial_lr}"]
    for epoch, lr in milestones.items():
        description.append(f"At epoch {epoch}: {lr}")
    return ", ".join(description)


@dataclass
class TrainingConfig:
    activation_function: callable  
    activation_function_name: str
    network_size: list
    filenames: list
    epochs: int
    epoch_switch: int
    data_points: int    

    # Equation scaling      
    U_c: float
    L_c: float
    n0_c: float
    # Weights for loss function
    weight_ic: float
    weight_bc: float
    weight_f: float
    weight_data: float

    # Learning rate scheduling
    initial_learning_rate: float = 1e-2
    schedule_milestones: dict = field(default_factory=lambda: {900_000: 5e-3, 1_500_000: 1e-3})
    lr_schedule: callable = field(init=False)

    def __post_init__(self):
        self.lr_schedule = optax.piecewise_constant_schedule(self.initial_learning_rate, self.schedule_milestones)

    @property
    def schedule_description(self):
        return create_schedule_description(self.initial_learning_rate, self.schedule_milestones)


config = TrainingConfig(
    activation_function = nn.sigmoid,
    activation_function_name=nn.sigmoid.__name__,
    network_size = [8, 8, 1],
    filenames = ['/Users/linus/Desktop/Github/Pinns-insulation-design/poisson/data/Case6_field.csv',
                 '/Users/linus/Desktop/Github/Pinns-insulation-design/poisson/data/Case6_Potential.csv',
                 '/Users/linus/Desktop/Github/Pinns-insulation-design/poisson/data/Case6_SpaceCharge.csv'],
    epochs = 100_000,
    epoch_switch = 300_000,
    data_points = 10,
    U_c = 1000,
    L_c = 0.01,
    n0_c = 1e18,
    weight_ic = 1e4,
    weight_bc = 1e4,
    weight_f = 1,
    weight_data = 1e2,
)

# Log everything in the start
# Now you can directly log the schedule description from the config instance
# Convert config to dict and log
logging.basicConfig(filename='training_log.log', level=logging.INFO, format='%(asctime)s - %(message)s', filemode='w')

config_dict = asdict(config)
logging.info('Training Configuration: %s', config_dict)
logging.info('Learning Rate Schedule: %s', config.schedule_description)



def plot_nn_predictions_fixed():
    """
    Plot predictions of potential and electric field from a neural network against true values using fixed paths and parameters.
    """
    # Assuming x_eval, uNN, electric_field_single, and params are predefined in your workspace
    
    # Define the paths to your CSV files
    field_path = config.filenames[0]
    potential_path = config.filenames[1]
    space_charge_path = config.filenames[2]
    # Generate evaluation points
    x_eval = np.linspace(0, 1, 100)[:, None]
    
    # Vectorize the electric field computation
    electric_field_batch = jax.jit(jax.vmap(electric_field_single, in_axes=(None, 0)))
    
    # Compute predicted potential and electric field
    nn_solution = uNN(params, jnp.array(x_eval)).reshape(-1)
    e_field_nn = electric_field_batch(params, jnp.array(x_eval)).reshape(-1)
    
    # Load true data from CSV files
    field_df = pd.read_csv(field_path, skiprows=7)
    x_data = field_df[['x-coordinate (m)']].values
    e_data = field_df[['Electric field norm']].values
    
    potential_df = pd.read_csv(potential_path, skiprows=7)
    potential_data = potential_df[['Electric potential']].values

    space_charge_df = pd.read_csv(space_charge_path, skiprows=7)
    space_charge_data = space_charge_df[['Space Charge Density']].values.flatten()
    n0_guess_plot = config.n0_c * current_n0 * (x_data - 0.005)**3
    
    
    # Plotting
    fig, axs = plt.subplots(1, 3, figsize=(15, 5))
    
    # Plot true vs predicted potential
    axs[0].plot(x_data, potential_data, label='True Potential U(x)', linestyle='-', color='blue')
    axs[0].plot(x_eval*config.L_c, (config.U_c)*nn_solution, label='Predicted Potential U(x)', linestyle='--', color='red')
    axs[0].set_xlabel('x (m)')
    axs[0].set_ylabel('U(x) (V)')
    axs[0].legend()
    axs[0].set_title('Potential U(x)')
    
    # Plot true vs predicted electric field
    axs[1].plot(x_data, e_data, label='True Electric Field E(x)', linestyle='-', color='blue')
    axs[1].plot(x_eval*config.L_c, (config.U_c/config.L_c)*e_field_nn, label="Predicted Electric Field E(x)", linestyle='--', color='red')
    axs[1].set_xlabel('x (m)')
    axs[1].set_ylabel("E(x) (V/m)")
    axs[1].legend()
    axs[1].set_title("Electric Field E(x)")

    # Plot true vs guess space charges
    axs[2].plot(x_data, space_charge_data, label='True Space Charges', linestyle='-', color='blue')
    axs[2].plot(x_data, n0_guess_plot.flatten(), label="Guess for n0 * (x - 0.5)^3", linestyle='--', color='red')
    axs[2].set_xlabel('x (m)')
    axs[2].set_ylabel("Space Charges")
    axs[2].legend()
    axs[2].set_title("Space Charges")
    
    plt.tight_layout()
    plt.show()


def generate_dataset(N=100, X=1.0, T=1.0):
    # Generate evenly spaced points for time and space
    t_vals = np.linspace(0, T, N).reshape(1, -1)  # 1 row, N columns for time
    x_vals = np.linspace(0, X, N).reshape(1, -1)  # 1 row, N columns for space

    # Stack them into a 2D array where the first row is t and the second row is x
    t_x_vals = np.vstack([t_vals, x_vals])

    return t_x_vals


class TimeDependentMLP(nn.Module):
    features: Sequence[int]

    def setup(self):
        self.layers = [nn.Dense(feat) for feat in self.features]

    def __call__(self, inputs):
        t, x = inputs[:, 0], inputs[:, 1]  # Correctly split the spatial and time components
        inputs = jnp.concatenate([x[:, None], t[:, None]], axis=1)  # Correct concatenation
        x = inputs
        for layer in self.layers[:-1]:
            x = config.activation_function(layer(x))
        return self.layers[-1](x)  # No activation after last layer, as we need a linear output
    
@jax.jit
def MSE(true, pred):
    return jnp.mean((true - pred) ** 2)


@jax.jit
def PINN_f(t, x, params):
    # Directly use uNN inside PINN_f, capturing `t` from the surrounding scope for differentiation
    # Ensure `t` is appropriately shaped for use in uNN, if necessary
    u_x = jax.grad(lambda x: uNN(params, t, x).squeeze(), argnums=0)(x)
    
    # For u_t, `t` is already the first argument to the lambda, so this should be fine
    u_t = jax.grad(lambda t: uNN(params, t, x).squeeze(), argnums=0)(t)
    
    # The residual of the PDE is u_t + u_x
    pde_residual = u_t + u_x
    
    return pde_residual


@jax.jit
def uNN(params, t, x):
    # Ensure t and x are reshaped into 2D arrays for concatenation
    t_reshaped = t.reshape(-1, 1)
    x_reshaped = x.reshape(-1, 1)
    inputs = jnp.concatenate([x_reshaped, t_reshaped], axis=1)
    u = model.apply(params, inputs)
    return u



@jax.jit
def loss_fun(params, t_x_domain, n_0, n_inj):
    # ufunc accepts both t and x
    ufunc = lambda t, x: uNN(params, t, x).squeeze()

    t_vals = t_x_domain[:, 0]
    x_vals = t_x_domain[:, 1]

    # Calculate the time derivative of u
    u_t = lambda t, x: jax.grad(lambda t: jnp.sum(ufunc(t, x)))(t)
    
    # Calculate the spatial derivative of u
    u_x = lambda t, x: jax.grad(lambda x: jnp.sum(ufunc(t, x)))(x)

    # Compute the physics-informed loss
    mse_f = jnp.mean((u_t(t_x_domain[:, 0:1], t_x_domain[:, 1:2]) +
                      u_x(t_x_domain[:, 0:1], t_x_domain[:, 1:2]))**2)

    # Calculate the loss for the initial condition at t=0
    t_0 = jnp.zeros_like(t_x_domain[:, 0:1])
    initial_condition_loss = MSE(ufunc(t_0, t_x_domain[:, 1:2]), n_0)

    # Calculate the loss for the spatial boundary condition at x=0
    x_0 = jnp.zeros_like(t_x_domain[:, 1:2])
    spatial_boundary_loss = MSE(u_x(t_x_domain[:, 0:1], x_0), n_inj)

    # Combine the losses
    total_loss = (config.weight_ic * initial_condition_loss + 
                  config.weight_bc * spatial_boundary_loss + 
                  config.weight_f * mse_f)
    
    return total_loss



@jax.jit
def update(opt_state, params, tx_domain, n_0, n_inj):
    grads = jax.grad(loss_fun, argnums=0)(params, tx_domain, n_0, n_inj)
    updates, opt_state = optimizer.update(grads, opt_state)
    params = optax.apply_updates(params, updates)
    return opt_state, params



def init_process(feats):
    model = TimeDependentMLP(features=feats)
    key1, key2 = jax.random.split(jax.random.PRNGKey(42), num=2)
    # Ensure the dummy input is a 2D array with shape (1, 2)
    dummy_in = jax.random.normal(key1, (1, 2))
    params = model.init(key2, dummy_in)
    lr_schedule = config.lr_schedule 
    optimizer = optax.adam(lr_schedule)
    opt_state = optimizer.init(params)

    return model, params, optimizer, opt_state



features = config.network_size # size of network
N_equation = config.data_points

CHARGE_GUESS = 100.0
# Normalized U-values
n_0 = 1.0  # Initial condition value for u(t=0)
n_inj = 0.0  # Boundary condition for u_x at x=0
filename = config.filenames[0]

tx_domain = generate_dataset(N=N_equation)
print(f"Data used:\n {tx_domain}")
print(f"Starting training")
model, params, optimizer, opt_state = init_process(features)
epochs = config.epochs

best_loss = float('inf')
best_params = None

t_0 = jnp.zeros_like(tx_domain[:, 0:1])
x_0 = jnp.zeros_like(tx_domain[:, 1:2])

ufunc = lambda t, x: uNN(params, t, x).squeeze()

# Vectorize PINN_f over the first dimension (batch dimension) of t and x inputs
# Adjust in_axes because ufunc is no longer an argument
batch_PINN_f = jax.vmap(PINN_f, in_axes=(0, 0, None))

t_vals, x_vals = tx_domain[:, 0], tx_domain[:, 1]
for epoch in range(epochs):
    opt_state, params = update(opt_state, params, tx_domain, n_0, n_inj)

    ufunc = lambda t, x: uNN(params, t, x).squeeze()

    if epoch % 1_000 == 0:
        #print(f"x-data: \n{x_data}")
        u_x = lambda x: jax.grad(lambda x: jnp.sum(ufunc(x)))(x)
        mse_f_vals = batch_PINN_f(t_vals, x_vals, params)  # params is used directly inside PINN_f
        mse_f = jnp.mean(mse_f_vals ** 2)
        #data_loss = MSE(normalized_output_field, e_data)
        # Compute boundary condition losses
        initial_condition_loss = MSE(ufunc(t_0, tx_domain[:, 1:2]), n_0)
        u_x = lambda t, x: jax.grad(lambda x: jnp.sum(ufunc(t, x)))(x)

        spatial_boundary_loss = MSE(u_x(tx_domain[:, 0:1], x_0), n_inj)

        
        total_loss = config.weight_ic * initial_condition_loss + config.weight_bc * spatial_boundary_loss + config.weight_f * mse_f

        
        # Assuming total_loss is calculated as shown in your code
        if total_loss < best_loss:
            best_loss = total_loss
            best_params = params  # Assuming 'params' holds your model parameters
            logging.info(f'New best model found at epoch {epoch} with loss {best_loss:.3e}')

        # Print the detailed losses
        print(f'Epoch = {epoch}, '
            f'Total Loss = {total_loss:.3e}, '
            f'DE Loss = {mse_f:.3e}, ')
            

        # Log the detailed losses instead of printing
        logging.info(f'Epoch = {epoch}, '
                     f'Total Loss = {total_loss:.3e}, '
                     f'DE Loss = {mse_f:.3e}, ')
                 
        
        #plot_nn_predictions_fixed()
print(f"Training complete")
# At the end of your training script
with open('best_model_params.pkl', 'wb') as f:
    pickle.dump(best_params, f)

logging.info('Training complete. Best model parameters saved to best_model_params.pkl')

# After your training loop
if best_params is not None:
    with open('best_model_params.pkl', 'wb') as f:
        pickle.dump(best_params, f)
    logging.info(f'Training complete. Best model parameters saved to best_model_params.pkl')
else:
    logging.info('Training complete. No improvement found.')
plot_nn_predictions_fixed()

# SOTA Total loss = 2.917e-06 (forward)
# With data loss
# Epoch = 990000, Total Loss = 2.117e-01, DE Loss = 2.610e-02, BC Loss 1 = 2.878e-13, BC Loss 2 = 1.283e-12, Data Loss = 1.856e-01

