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


def plot_nn_predictions_fixed():
    """
    Plot predictions of potential and electric field from a neural network against true values using fixed paths and parameters.
    """
    # Assuming x_eval, uNN, electric_field_single, and params are predefined in your workspace
    
    # Define the paths to your CSV files
    field_path = '/Users/linus/Desktop/Github/Pinns-insulation-design/poisson/data/Case3_field.csv'
    potential_path = '/Users/linus/Desktop/Github/Pinns-insulation-design/poisson/data/Case3_Potential.csv'
    
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
    
    # Plotting
    fig, axs = plt.subplots(1, 2, figsize=(15, 5))
    
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
    n0 = 1e16
    L_c = 0.01
    U_c = 10
   
    u_x = lambda x: jax.grad(lambda x: jnp.sum(ufunc(x)))(x)
    u_xx = lambda x: jax.grad(lambda x: jnp.sum(u_x(x)))(x)
    return epsilon*u_xx(x)*U_c/((L_c**5)*q*n0) +(x-0.5)**3
    
@jax.jit
def uNN(params, x):
    u = model.apply(params, x)
    return u


@jax.jit
def loss_fun(params, data_fitting, data_de, U_0, U_1):
    ufunc = lambda x: uNN(params, x).squeeze()  # Ensure this is scalar for each x
    u_x = lambda x: jax.grad(lambda x: jnp.sum(ufunc(x)))(x)
    #e_field_data = (e_field_data - E_min) / (E_max - E_min)
    normalized_output_field = -1000*u_x(x_data)
    #print(f"Output before min-max:\n {normalized_output_field}")
    normalized_output_field = (normalized_output_field - E_min) / (E_max - E_min)
    #print(f"Output after min-max:\n {normalized_output_field}")
    # Compute DE loss using DE data
    mse_f = jnp.mean(PINN_f(data, ufunc, params) ** 2)
    #data_loss = MSE(normalized_output_field, e_data)
    # Compute boundary condition losses
    bc_loss1 = MSE(ufunc(jnp.array([[0.0]])), U_0)
    bc_loss2 = MSE(ufunc(jnp.array([[1.0]])), U_1)

    #mse_data = MSE(normalized_output_field, e_data)
    total_loss = mse_f + 1e4*bc_loss1 + 1e4*bc_loss2
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
    lr = optax.piecewise_constant_schedule(1e-2, {40_000: 5e-3, 750_000: 1e-3})
    optimizer = optax.adam(lr)
    opt_state = optimizer.init(params)

    return model, params, optimizer, opt_state


features = [128, 128, 1] # size of network

N_data = 100 # number of sampled points
N_equation = 100

CHARGE_GUESS = 5.0*10**4
# Normalized U-values
U_0 = 1
U_1 = 0
filename = 'poisson/data/Case3_Field.csv'

data = generate_dataset(N=100)
e_field_data, E_min, E_max = load_electric_field(filename)
x_data, e_data = e_field_data[:, [0]]/0.01, e_field_data[:, [1]]
print(f"For this run, we have: E_min = {E_min} and E_max = {E_max}")
print(f"Starting training")
model, params, optimizer, opt_state = init_process(features, CHARGE_GUESS)
epochs = 1_000_000
for epoch in range(epochs):
    opt_state, params = update(opt_state, params, e_field_data, data, U_0, U_1)
    
    if epoch % 10_000 == 0:
        

        #print(f"x-data: \n{x_data}")
        ufunc = lambda x: uNN(params, x).squeeze()  # Ensure this is scalar for each x
        u_x = lambda x: jax.grad(lambda x: jnp.sum(ufunc(x)))(x)
        #e_field_data = (e_field_data - E_min) / (E_max - E_min)
        normalized_output_field = -1000*u_x(x_data)
        #print(f"Output before min-max:\n {normalized_output_field}")
        normalized_output_field = (normalized_output_field - E_min) / (E_max - E_min)
        #print(f"Output after min-max:\n {normalized_output_field}")
        # Compute DE loss using DE data
        mse_f = jnp.mean(PINN_f(data, ufunc, params) ** 2)
        #data_loss = MSE(normalized_output_field, e_data)
        # Compute boundary condition losses
        bc_loss1 = MSE(ufunc(jnp.array([[0.0]])), U_0)
        bc_loss2 = MSE(ufunc(jnp.array([[1.0]])), U_1)

        mse_data = MSE(normalized_output_field, e_data)
        total_loss = mse_f + 1e4*bc_loss1 + 1e4*bc_loss2 + mse_data
        current_charge = params["params"]["charge"][0]

        # Print the detailed losses
        print(f'Epoch = {epoch}, '
            f'Total Loss = {total_loss:.3e}, '
            f'DE Loss = {mse_f:.3e}, '
            f'BC Loss 1 = {bc_loss1:.3e}, '
            f'BC Loss 2 = {bc_loss2:.3e}, '
            f'Data Loss = {mse_data:.3e}')

        
        #plot_nn_predictions_fixed()
print(f"Training complete")

plot_nn_predictions_fixed()

# SOTA Total loss = 2.917e-06 (forward)
# With data loss
# Epoch = 990000, Total Loss = 2.117e-01, DE Loss = 2.610e-02, BC Loss 1 = 2.878e-13, BC Loss 2 = 1.283e-12, Data Loss = 1.856e-01

