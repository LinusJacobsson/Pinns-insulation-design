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

@jax.jit
def electric_field_single(params, x):
    # Extract only the potential from the outputs of uNN
    # The lambda function needs to ensure it operates on a scalar value
    dU_dx = jax.grad(lambda x: uNN(params, x)[0].squeeze())(x)
    
    # Return the negative of the gradient to represent the electric field
    # No additional squeeze should be needed if dU_dx is correctly computed as a scalar
    return -dU_dx


def plot_nn_predictions_fixed():
    """
    Plot predictions of potential and electric field from a neural network against true values using fixed paths and parameters.
    """
    # Assuming x_eval, uNN, electric_field_single, and params are predefined in your workspace
    
    # Define the paths to your CSV files
    field_path = '/Users/linus/Desktop/Github/Pinns-insulation-design/poisson/data/Case3_field.csv'
    potential_path = '/Users/linus/Desktop/Github/Pinns-insulation-design/poisson/data/Case3_Potential.csv'
    space_charge_path = '/Users/linus/Desktop/Github/Pinns-insulation-design/poisson/data/Case3_SpaceCharge.csv'
    # Generate evaluation points
    x_eval = np.linspace(0, 1, 301)[:, None]
    
    # Vectorize the electric field computation
    electric_field_batch = jax.jit(jax.vmap(electric_field_single, in_axes=(None, 0)))
    
    # Compute predicted potential and electric field
    nn_solution = uNN(params, jnp.array(x_eval))[0].reshape(-1)
    e_field_nn = electric_field_batch(params, jnp.array(x_eval)).reshape(-1)
    
    # Load true data from CSV files
    field_df = pd.read_csv(field_path, skiprows=7)
    x_data = field_df[['x-coordinate (m)']].values
    e_data = field_df[['Electric field norm']].values
    
    potential_df = pd.read_csv(potential_path, skiprows=7)
    potential_data = potential_df[['Electric potential']].values

    space_charge_df = pd.read_csv(space_charge_path, skiprows=7)
    space_charge_data = space_charge_df[['Space Charge Density']].values.flatten()
    _, space_charge_guess = uNN(params, jnp.array(x_eval))
    print(f'Predicted space charge distribution:\n {space_charge_guess}')
    
    # Plotting
    fig, axs = plt.subplots(1, 3, figsize=(15, 5))
    
    # Plot true vs predicted potential
    axs[0].plot(x_data, potential_data, label='True Potential U(x)', linestyle='-', color='blue')
    axs[0].plot(x_eval/100, 1e1*nn_solution, label='Predicted Potential U(x)', linestyle='--', color='red')
    axs[0].set_xlabel('x (m)')
    axs[0].set_ylabel('U(x) (V)')
    axs[0].legend()
    axs[0].set_title('Potential U(x)')
    
    # Plot true vs predicted electric field
    axs[1].plot(x_data, e_data, label='True Electric Field E(x)', linestyle='-', color='blue')
    axs[1].plot(x_eval/100, 1e3*e_field_nn, label="Predicted Electric Field E(x)", linestyle='--', color='red')
    axs[1].set_xlabel('x (m)')
    axs[1].set_ylabel("E(x) (V/m)")
    axs[1].legend()
    axs[1].set_title("Electric Field E(x)")

    # Plot true vs guess space charges
    #axs[2].plot(x_data, space_charge_data, label='True Space Charges', linestyle='-', color='blue')
    axs[2].plot(x_data, space_charge_guess.flatten(), label="Guess for n0 * (x - 0.5)^3", linestyle='--', color='red')
    axs[2].set_xlabel('x (m)')
    axs[2].set_ylabel("Space Charges")
    axs[2].legend()
    axs[2].set_title("Space Charges")
    
    plt.tight_layout()
    plt.show()

def load_electric_field(filename, normalize = True):
    df = pd.read_csv(filename, skiprows=8, header=None, names=['x-coordinate (m)', 'Electric field norm'])
    dataset = df.values
    
    # Normalize x values to the range [0, 1]
    #x_min, x_max = np.min(dataset[:, 0]), np.max(dataset[:, 0])
    #dataset[:, 0] = (dataset[:, 0] - x_min) / (x_max - x_min)
    
    if normalize:
        # Normalize Electric field values to the range [0, 1]
        E_min, E_max = np.min(dataset[:, 1]), np.max(dataset[:, 1])
        print(f"E_min: {E_min}")
        print(f"E_max: {E_max}")
        dataset[:, 1] = (dataset[:, 1] - E_min) / (E_max - E_min)
        
    return dataset, E_min, E_max

def generate_dataset(N=100, noise_percent=0.0, seed=420, charge = 100):
   
    np.random.seed(seed)
    xmin, xmax = 0.0, 1.0
    x_vals = np.random.uniform(low=xmin, high=xmax, size=(N, 1))
    return x_vals

class CombinedModel(nn.Module):
    features: Sequence[int]

    def setup(self):
        self.dense_layers = [nn.Dense(feat) for feat in self.features[:-1]]
        # Output layer for potential U and space charge distribution
        self.output_layer = nn.Dense(2)  # Adjust for 2 outputs

    def __call__(self, x):
        for layer in self.dense_layers:
            x = nn.tanh(layer(x))
        return self.output_layer(x)  # This will output both U and space charge

    
@jax.jit
def MSE(true, pred):
    return jnp.mean((true - pred) ** 2)


def PINN_f(x, params, include_data_loss):
    epsilon = 2*8.85e-12
    q = 1.6e-19
    L_c = 0.01
    U_c = 10

    potential, space_charge = uNN(params, x)  # uNN now returns both values

    # Gradient and second derivative of the potential
    u_x = jax.grad(lambda x: jnp.sum(potential))(x)
    u_xx = jax.grad(lambda x: jnp.sum(u_x))(x)

    # Adjust the equation to use the predicted space charge directly
    # Note: Ensure space_charge is appropriately scaled or transformed if necessary
    pde_residual = epsilon * u_xx * U_c / ((L_c ** 5)) + space_charge

    return pde_residual

@jax.jit
def uNN(params, x):
    # Ensure x is always 2D: [batch_size, features]
    x = x if x.ndim == 2 else x[None, :]
    outputs = model.apply(params, x)
    if outputs.ndim == 1:
        outputs = outputs[None, :]  # Add batch dimension if missing
    potential, space_charge = outputs[:, 0], outputs[:, 1]
    return potential, space_charge


@jax.jit
def loss_fun(params, data_fitting, data_equation, U_0, U_1, include_data_loss):
    # Adjusted to unpack the two outputs: potential and space charge
    ufunc = lambda x: uNN(params, x)[0].squeeze()  # Assuming the first output is the potential
    u_x = lambda x: jax.grad(lambda x: jnp.sum(ufunc(x)))(x)
    normalized_output_field = -1e3*u_x(data_fitting[:, [0]]/0.01)
    normalized_output_field = (normalized_output_field - E_min) / (E_max - E_min)

    def true_fun(_):
        return MSE(normalized_output_field, data_fitting[:, [1]])
    
    def false_fun(_):
        return 0.0

    data_loss = lax.cond(include_data_loss, true_fun, false_fun, None)

    # Compute PDE residual using the updated `uNN` which now outputs both potential and space charge
    mse_f = jnp.mean(PINN_f(data_equation, params, include_data_loss) ** 2)

    # For boundary conditions, extract and use the potential output
    potential_at_0, _ = uNN(params, jnp.array([[0.0]]))  # Unpack potential and ignore space charge
    potential_at_1, _ = uNN(params, jnp.array([[1.0]]))  # Unpack potential and ignore space charge
    bc_loss1 = MSE(potential_at_0, U_0)
    bc_loss2 = MSE(potential_at_1, U_1)
    
    total_loss = 1e4 * bc_loss1 + 1e4 * bc_loss2 + 1e4 * data_loss + mse_f
    
    return total_loss



@jax.jit
def update(opt_state, params, data_fitting, data_equation, U_0, U_, include_data_loss):
    # Get the gradient w.r.t to MLP params
    grads = jax.jit(jax.grad(loss_fun, 0))(params, data_fitting, data_equation, U_0, U_1, include_data_loss)
    # Update params
    updates, opt_state = optimizer.update(grads, opt_state)
    params = optax.apply_updates(params, updates)

    return opt_state, params


def init_process(feats):
    model = CombinedModel(features=feats)
    key1, key2 = jax.random.split(jax.random.PRNGKey(420), num=2)
    dummy_in = jax.random.normal(key1, (1,))
    params = model.init(key2, dummy_in)
    lr = optax.piecewise_constant_schedule(1e-2, {40_000: 5e-3, 2_500_000: 1e-3})
    optimizer = optax.adam(lr)
    opt_state = optimizer.init(params)

    return model, params, optimizer, opt_state


features = [8, 8, 1] # size of network

N_data = 100 # number of sampled points
N_equation = 100

CHARGE_GUESS = 1.0
# Normalized U-values
U_0 = 1
U_1 = 0
filename = 'poisson/data/Case3_Field.csv'

data_equation = generate_dataset(N=100)
e_field_data, E_min, E_max = load_electric_field(filename)
x_data, e_data = e_field_data[:, [0]]/0.01, e_field_data[:, [1]]
print(f"For this run, we have: E_min = {E_min} and E_max = {E_max}")
print(f"Starting training")
model, params, optimizer, opt_state = init_process(features)
epochs = 3_000_000
switch_epoch = 300_000  # Define when to start including data loss

best_loss = float('inf')
best_params = None
for epoch in range(epochs):
    include_data_loss = epoch >= switch_epoch
    opt_state, params = update(opt_state, params, e_field_data, data_equation, U_0, U_1, include_data_loss)
    
    if epoch % 10_000 == 0:
        

        #print(f"x-data: \n{x_data}")
        ufunc = lambda x: uNN(params, x)[0].squeeze()  # Assuming the first output is the potential
        u_x = lambda x: jax.grad(lambda x: jnp.sum(ufunc(x)))(x)
        #e_field_data = (e_field_data - E_min) / (E_max - E_min)
        normalized_output_field = -1e3*u_x(x_data)
        #print(f"Output before min-max:\n {normalized_output_field}")
        normalized_output_field = (normalized_output_field - E_min) / (E_max - E_min)
        #print(f"Output after min-max:\n {normalized_output_field}")
        # Compute DE loss using DE data
        mse_f = jnp.mean(PINN_f(data_equation, params, include_data_loss) ** 2)
        #data_loss = MSE(normalized_output_field, e_data)
        # Compute boundary condition losses
        bc_loss1 = MSE(ufunc(jnp.array([[0.0]])), U_0)
        bc_loss2 = MSE(ufunc(jnp.array([[1.0]])), U_1)

        mse_data = MSE(normalized_output_field, e_data)
        if epoch < switch_epoch:
            mse_data = 0
        total_loss = mse_f + 1e4*bc_loss1 + 1e4*bc_loss2 + mse_data
        

        

        # Print the detailed losses
        print(f'Epoch = {epoch}, '
            f'Total Loss = {total_loss:.3e}, '
            f'DE Loss = {mse_f:.3e}, '
            f'BC Loss 1 = {bc_loss1:.3e}, '
            f'BC Loss 2 = {bc_loss2:.3e}, '
            f'Data Loss = {mse_data:.3e}, ')

        
        #plot_nn_predictions_fixed()
print(f"Training complete")

plot_nn_predictions_fixed()

# SOTA Total loss = 2.917e-06 (forward)
# With data loss
# Epoch = 990000, Total Loss = 2.117e-01, DE Loss = 2.610e-02, BC Loss 1 = 2.878e-13, BC Loss 2 = 1.283e-12, Data Loss = 1.856e-01
