import jax 
import jax.numpy as jnp
import numpy as np
import optax
import flax.linen as nn
from typing import Sequence, Callable
import matplotlib.pyplot as plt


"""@jax.jit
def potential(x, const):
    return -(const*x**3)/6 + (x/15)*(250*const-189) + 1

@jax.jit
def electric_field(x, const):
    return -(const * x**2) / 2 + (250*const-189)/15"""

def potential(x, const):
    return -(const*x**3)/6 + (const/6 - 11)*x + 1

def electric_field(x, const):
    return -(const * x**2)/2 + (const/6 - 11)

@jax.jit
def electric_field_single(params, x):
    # Compute the gradient of the neural network output with respect to its input
    dU_dx = jax.grad(lambda x: jnp.squeeze(uNN(params, x)))(x)
    
    # Return the negative of the gradient to represent the electric field
    return dU_dx


def generate_dataset(N=10, noise_percent=0.0, seed=420, charge = 1000):
    np.random.seed(seed)
    xmin, xmax = 0.0, 1.0
    # Uniform random
    #x_vals = np.random.uniform(low=xmin, high=xmax, size=(N, 1))
    x_vals = np.linspace(xmin, xmax, num=N).reshape(-1, 1)  # Correctly generate N data points
    u_vals = electric_field(x=x_vals, const=charge)
    #noise = np.random.normal(0, u_vals.std(), [N, 1]) * noise_percent
    #u_vals += noise
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
        for idx, layer in enumerate(self.layers):
            x = layer(x)
            if idx != len(self.layers)-1:
                x = jnp.tanh(x)
        return x
@jax.jit
def MSE(true, pred):
    return jnp.mean((true - pred) ** 2)
    

def PINN_f(x, ufunc, params):
    charge = params["params"]["charge"][0]
    u_x = lambda x: jax.grad(lambda x: jnp.sum(ufunc(x)))(x)
    u_xx = lambda x: jax.grad(lambda x: jnp.sum(u_x(x)))(x)
    return u_xx(x)/charge + x

@jax.jit
def uNN(params, x):
    u = model.apply(params, x)
    return u


@jax.jit
def loss_fun(params, data_fitting, data_de, xmin, xmax, U_0, U_1):
    # Split data fitting for data loss calculation
    x_data, u_data = data_fitting[:, [0]], data_fitting[:, [1]]
    ufunc = lambda x: uNN(params, x)

    # Compute data loss as before
    du_dx = lambda x: jax.grad(lambda x: jnp.sum(ufunc(x)))(x)
    data_loss = MSE(du_dx(x_data), u_data)

    # Split DE data for function loss calculation
    x_de, _ = data_de[:, [0]], data_de[:, [1]]  # Assuming DE data has the same format

    # Compute DE loss (function_loss or mse_f) using x_de
    mse_f = jnp.mean(PINN_f(x_de, ufunc, params) ** 2)

    # Compute boundary condition losses as before
    bc_loss1 = MSE(ufunc(jnp.array([[xmin]])), U_0)
    bc_loss2 = MSE(ufunc(jnp.array([[xmax]])), U_1)

    # Combine losses
    total_loss = 100*mse_f + bc_loss1 + bc_loss2 + 100*data_loss
    return total_loss


@jax.jit
def update(opt_state, params, data_fitting, data_equation, U_0, U_1):
    # Get the gradient w.r.t to MLP params
    grads = jax.jit(jax.grad(loss_fun, 0))(params, data_fitting, data_equation, xmin, xmax, U_0, U_1)

    # Update params
    updates, opt_state = optimizer.update(grads, opt_state)
    params = optax.apply_updates(params, updates)

    return opt_state, params


def init_process(feats, charge_guess):
    model = MLP(features=feats, charge_value=charge_guess)

    key1, key2 = jax.random.split(jax.random.PRNGKey(420), num=2)

    dummy_in = jax.random.normal(key1, (1,))
    params = model.init(key2, dummy_in)

    lr = optax.piecewise_constant_schedule(1e-2, {50_000: 5e-3, 80_000: 1e-3})
    optimizer = optax.adam(lr)
    opt_state = optimizer.init(params)

    return model, params, optimizer, opt_state



features = [8, 8, 1] # size of network

N_data = 10 # number of sampled points
N_equation = 10

CHARGE = 10.0 # Just nu funkar värden mellan 1e-2 till 1e0 utan ändringar 

CHARGE_GUESS = 300.0

U_0 = 1
U_1 = -10

data_fitting, xmin, xmax = generate_dataset(N=N_data, charge=CHARGE)
data_equation, _, _ = generate_dataset(N=N_equation, charge=CHARGE)  # Assuming generate_dataset can accept noise_percent=0.0 to generate without noise


model, params, optimizer, opt_state = init_process(features, CHARGE_GUESS)
epochs = 100_000
for epoch in range(epochs):
    opt_state, params = update(opt_state, params, data_fitting, data_equation, U_0, U_1)
    
        # Conditionally log detailed loss components every 1000th epoch
    if epoch % 1000 == 0:
        # Split datasets for logging
        x_data_fitting, u_data_fitting = data_fitting[:, [0]], data_fitting[:, [1]]
        x_de = data_equation[:, [0]]  # Assuming DE data format matches

        ufunc = lambda x: uNN(params, x)

        # Compute gradients and losses for data fitting
        du_dx_data_fitting = lambda x: jax.grad(lambda x: jnp.sum(ufunc(x)))(x)
        data_loss = MSE(du_dx_data_fitting(x_data_fitting), u_data_fitting)

        # Compute DE loss using DE data
        mse_f = jnp.mean(PINN_f(x_de, ufunc, params) ** 2)

        # Compute boundary condition losses
        bc_loss1 = MSE(ufunc(jnp.array([[xmin]])), U_0)
        bc_loss2 = MSE(ufunc(jnp.array([[xmax]])), U_1)

        # Combine losses for total loss
        total_loss = 100*mse_f + bc_loss1 + bc_loss2 + 100*data_loss

        # Print the detailed losses
        print(f'Epoch = {epoch}, Total Loss = {total_loss:.3e}, DE Loss = {mse_f:.3e}, BC Loss 1 = {bc_loss1:.3e}, BC Loss 2 = {bc_loss2:.3e}, Data Loss = {data_loss:.3e}')


current_charge = params["params"]["charge"][0]

# Generate a set of time points for evaluation
x_eval = np.linspace(xmin, xmax, 500)[:, None]

# Compute the analytical solution
solution = potential(x=x_eval, const=CHARGE)

# Compute the neural network prediction
nn_solution = uNN(params, jnp.array(x_eval))

# Vectorize the electric_field_single function to work over batches of inputs
electric_field_batch = jax.jit(jax.vmap(electric_field_single, in_axes=(None, 0)))

# Now use electric_field_batch to compute the electric field for all points in x_eval
e_field_nn = electric_field_batch(params, jnp.array(x_eval).reshape(-1, 1))
# Evaluate the analytical electric field over the same range


analytical_e_field = electric_field(x_eval, const=CHARGE)

# Your existing plotting code
fig = plt.figure(figsize=(10, 6))  # Store the figure in a variable

# Plot the potential
plt.subplot(2, 1, 1)
plt.plot(x_eval, solution, label='Analytical Solution', color='blue')
plt.plot(x_eval, nn_solution, label='NN Prediction', linestyle='--', color='red')
plt.xlabel('x')
plt.ylabel('U(x)')
plt.title(f'True charge: {CHARGE:.4f}, Predicted charge: {current_charge:.4f}, Error: {100*(CHARGE-current_charge)/CHARGE:.2f}%')
plt.legend()
plt.grid()

# Plot the electric fields
plt.subplot(2, 1, 2)
plt.plot(x_eval, analytical_e_field, label='Analytical Electric Field', color='blue')
plt.scatter(data_fitting[:, 0], electric_field(data_fitting[:, 0], const=CHARGE), color='red', label='Training Data')
plt.plot(x_eval, e_field_nn, label='NN Predicted Electric Field', linestyle='--', color='red')
plt.xlabel('x')
plt.ylabel('E(x)')
#plt.ylim(-100, 100)
plt.title('Electric Field E(x)')
plt.legend()
plt.grid()

plt.tight_layout()

# Add an overall title to the figure
fig.suptitle('Overall Title for the Figure', fontsize=16, y=1.05)  # Adjust the font size and position as needed

plt.show()