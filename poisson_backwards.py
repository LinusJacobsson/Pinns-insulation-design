import jax 
import jax.numpy as jnp
import numpy as np
import optax
import flax.linen as nn
from typing import Sequence, Callable
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp

def solve_ode_for_x(x_values, k, x0, charge):
    # Define the ODE using the sigmoid function
    def sigmoid_ode(x, y):
        y0, y1 = y
        dydx = [y1, -charge * sigmoid(x, k=k, x0=x0)]
        return dydx
    
    # Initial conditions
    U0 = 1000
    U_prime_0 = 0
    
    # Solve the ODE
    x_span = [np.min(x_values), np.max(x_values)]
    y_init = [U0, U_prime_0]
    sol = solve_ivp(sigmoid_ode, x_span, y_init, t_eval=x_values)
    
    return sol

@jax.jit
def electric_field_single(params, x):
    # Compute the gradient of the neural network output with respect to its input
    dU_dx = jax.grad(lambda x: jnp.squeeze(uNN(params, x)))(x)
    
    # Return the negative of the gradient to represent the electric field
    return dU_dx

# Your sigmoid function
def sigmoid(x, k=10, x0=0.5):
    return 1 / (1 + jnp.exp(-k * (x - x0)))

def generate_dataset(N=10, noise_percent=0.0, seed=420, k=10.0, x0=0.5, charge = 1):
    np.random.seed(seed)
    xmin, xmax = 0.0, 0.5

    x_vals = np.linspace(xmin, xmax, num=N).reshape(-1, 1)  # x values where we want the ODE solution

    # ODE system using the sigmoid function
    def sigmoid_ode(x, y):
        y0, y1 = y
        dydx = [y1, -charge * sigmoid(x, k=k, x0=x0)]
        return dydx

    # Initial conditions
    U0 = 1000  # Initial condition for U
    U_prime_0 = 0  # Initial condition for U'

    # Solve the ODE
    x_span = [xmin, xmax]  # Interval of integration
    y_init = [U0, U_prime_0]

    sol = solve_ivp(sigmoid_ode, x_span, y_init, t_eval=x_vals.flatten())

    # Extract the solution for U at x_vals
    u_vals = sol.y[1].reshape(-1, 1)  # Reshape to match x_vals shape

    # Optionally add noise to the solution
    if noise_percent > 0.0:
        noise = np.random.normal(0, np.std(u_vals), u_vals.shape) * noise_percent
        u_vals += noise

    colloc = np.concatenate([x_vals, u_vals], axis=1)

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
    return u_xx(x) + charge * sigmoid(x)

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

    # Compute the first boundary condition loss for U at x = xmin
    bc_loss1 = MSE(ufunc(jnp.array([[xmin]])), U_0)
    
    # Compute the second boundary condition loss for y' = 0 at x = xmin
    du_dx_xmin = jax.grad(lambda x: jnp.sum(ufunc(jnp.array([[x]]))))(xmin)
    bc_loss2 = MSE(du_dx_xmin, U_1)  # Ensure the derivative at xmin is close to zero

    # Combine losses
    total_loss = 100*mse_f + 100*bc_loss1 + 10*bc_loss2 + 1000*data_loss
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

    lr = optax.piecewise_constant_schedule(1e-2, {80_000: 5e-3, 120_000: 1e-3})
    optimizer = optax.adam(lr)
    opt_state = optimizer.init(params)

    return model, params, optimizer, opt_state



features = [16, 16, 1] # size of network

N_data = 50 # number of sampled points
N_equation = 100 

CHARGE = 1_000.0 # Just nu funkar värden mellan 1e-2 till 1e0 utan ändringar 

CHARGE_GUESS = 900.0

U_0 = 1000
U_1 = 0

data_fitting, xmin, xmax = generate_dataset(N=N_data, charge=CHARGE)
data_equation, _, _ = generate_dataset(N=N_equation, charge=CHARGE)  # Assuming generate_dataset can accept noise_percent=0.0 to generate without noise

print(f"Starting training")
model, params, optimizer, opt_state = init_process(features, CHARGE_GUESS)
epochs = 100_000
for epoch in range(epochs):
    opt_state, params = update(opt_state, params, data_fitting, data_equation, U_0, U_1)
    
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
        # Compute the second boundary condition loss for y' = 0 at x = xmin
        du_dx_xmin = jax.grad(lambda x: jnp.sum(ufunc(jnp.array([[x]]))))(xmin)
        bc_loss2 = MSE(du_dx_xmin, 0.0)  # Ensure the derivative at xmin is close to zero
        # Combine losses for total loss
        total_loss = 1000*mse_f + bc_loss1 + bc_loss2 + 100*data_loss

        # Print the detailed losses
        print(f'Epoch = {epoch}, Total Loss = {total_loss:.3e}, DE Loss = {mse_f:.3e}, BC Loss 1 = {bc_loss1:.3e}, BC Loss 2 = {bc_loss2:.3e}, Data Loss = {data_loss:.3e}')

print(f"Training complete")
current_charge = params["params"]["charge"][0]

# Assuming x_eval is a numpy array of evaluation points
x_eval = np.linspace(xmin, xmax, 500)[:, None]
# Vectorize the electric_field_single function to work over batches of inputs
electric_field_batch = jax.jit(jax.vmap(electric_field_single, in_axes=(None, 0)))
# Compute predicted potential and electric field using your neural network

nn_solution = uNN(params, jnp.array(x_eval)).reshape(-1)
e_field_nn = electric_field_batch(params, jnp.array(x_eval)).reshape(-1)


# Assume sol is the object returned by solve_ivp
sol = solve_ode_for_x(x_eval.flatten(), k=10, x0=0.5, charge=CHARGE)

# Extract the solution for U(x) and its derivative U'(x)
U_x_true = sol.y[0]  # The first row of sol.y contains U(x)
E_x_true = sol.y[1]  # The second row of sol.y contains U'(x)

# Now, plot as before, but using these true values
fig, axs = plt.subplots(1, 3, figsize=(15, 5))

# Plot true vs predicted potential
axs[0].plot(x_eval, U_x_true, label='True Potential U(x)', color='blue')
axs[0].plot(x_eval, nn_solution, label='Predicted Potential U(x)', linestyle='--', color='red')
axs[0].set_xlabel('x')
axs[0].set_ylabel('U(x)')
axs[0].legend()
axs[0].set_title('Potential U(x)')

# Plot true vs predicted electric field
axs[1].plot(x_eval, E_x_true, label='True Electric Field U\'(x)', color='blue')
axs[1].scatter(data_fitting[:, 0], data_fitting[:, 1], color='k', label='Training Data')
axs[1].plot(x_eval, e_field_nn, label="Predicted Electric Field U'(x)", linestyle='--', color='red')
axs[1].set_xlabel('x')
axs[1].set_ylabel("U'(x)")
axs[1].legend()
axs[1].set_title("Electric Field U'(x)")

# Plot the sigmoid function n(x)
axs[2].plot(x_eval, CHARGE*sigmoid(x_eval, k=10, x0=0.5), label='Sigmoid Function n(x)', color='green')
axs[2].plot(x_eval, current_charge*sigmoid(x_eval, k=10, x0=0.5),linestyle='--', label='Predicted Sigmoid', color='red')
axs[2].set_xlabel('x')
axs[2].set_ylabel('n(x)')
axs[2].legend()
axs[2].set_title(f'True charge:{CHARGE}, Predicted charge:{current_charge:.2f}')

plt.tight_layout()
plt.show()
