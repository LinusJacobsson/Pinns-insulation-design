import jax.numpy as jnp
from jax import grad, jit, value_and_grad, vmap
import jax
from flax import linen as nn
import optax
import numpy as np
import matplotlib.pyplot as plt

class SolutionNN(nn.Module):
    @nn.compact
    def __call__(self, x):
        x = nn.Dense(64)(x)
        x = nn.relu(x)
        x = nn.Dense(1)(x)
        return x

class FunctionNN(nn.Module):
    @nn.compact
    def __call__(self, x):
        x = nn.Dense(64)(x)
        x = nn.relu(x)
        x = nn.Dense(1)(x)
        return x

def init_model():
    solution_model = SolutionNN()
    function_model = FunctionNN()
    rng = jax.random.PRNGKey(0)
    x_dummy = jnp.ones((1, 1))
    solution_params = solution_model.init(rng, x_dummy)['params']
    function_params = function_model.init(rng, x_dummy)['params']
    solution_optimizer = optax.adam(1e-3)
    function_optimizer = optax.adam(1e-3)
    solution_opt_state = solution_optimizer.init(solution_params)
    function_opt_state = function_optimizer.init(function_params)
    return solution_model, function_model, solution_params, function_params, solution_optimizer, function_optimizer, solution_opt_state, function_opt_state


def loss_fn(solution_params, function_params, x, y_true, solution_model, function_model):
    y_pred = solution_model.apply({'params': solution_params}, x)
    data_loss = jnp.mean((y_pred - y_true)**2)
    dy_dx = grad(lambda x: solution_model.apply({'params': solution_params}, x).squeeze())(x)
    f_x_pred = function_model.apply({'params': function_params}, x).squeeze()
    ode_loss = jnp.mean((dy_dx + f_x_pred)**2)
    total_loss = data_loss + ode_loss
    return total_loss


def update(solution_params, function_params, x, y_true, solution_opt_state, function_opt_state, solution_optimizer, function_optimizer):
    # Calculate gradients for both models
    solution_loss, solution_grads = jax.value_and_grad(loss_fn, argnums=0)(solution_params, function_params, x, y_true,solution_opt_state, function_opt_state)
    function_loss, function_grads = jax.value_and_grad(loss_fn, argnums=1)(solution_params, function_params, x, y_true,solution_opt_state, function_opt_state)
    
    # Apply updates to both models
    solution_updates, solution_opt_state = solution_optimizer.update(solution_grads, solution_opt_state, solution_params)
    solution_params = optax.apply_updates(solution_params, solution_updates)
    
    function_updates, function_opt_state = function_optimizer.update(function_grads, function_opt_state, function_params)
    function_params = optax.apply_updates(function_params, function_updates)
    
    # Combined loss for logging or debugging
    total_loss = solution_loss + function_loss
    
    return solution_params, function_params, solution_opt_state, function_opt_state, total_loss


def generate_data(N=100):
    x = jnp.linspace(-1, 1, N).reshape(-1, 1)
    y_true = x ** 2  # Example function y = x^2
    return x, y_true

# Initialize models, optimizers, and states
solution_model, function_model, solution_params, function_params, solution_optimizer, function_optimizer, solution_opt_state, function_opt_state = init_model()

# Generate synthetic data
x_train, y_train = generate_data(N=100)

# Training loop
epochs = 1000
for epoch in range(epochs):
    solution_params, function_params, solution_opt_state, function_opt_state, loss = update(
        solution_params, function_params, x_train, y_train, solution_opt_state, function_opt_state, solution_optimizer, function_optimizer
    )
    if epoch % 100 == 0:
        print(f"Epoch {epoch}, Loss: {loss}")


# Generate a range of x values for evaluation
x_eval = jnp.linspace(-1, 1, 100).reshape(-1, 1)

# Predict y using the solution model
y_pred = solution_model.apply({'params': solution_params}, x_eval)

# Predict f(x) using the function model
f_x_pred = function_model.apply({'params': function_params}, x_eval)

# True values for comparison
y_true = x_eval ** 2  # Assuming the true solution is y = x^2
f_x_true = 2 * x_eval  # Assuming the true f(x) is the derivative of x^2, which is 2x

# Plotting
plt.figure(figsize=(12, 5))

# Plot y and its prediction
plt.subplot(1, 2, 1)
plt.plot(x_eval, y_true, label='True y(x) = x^2', color='blue')
plt.plot(x_eval, y_pred, label='Predicted y(x)', linestyle='--', color='red')
plt.xlabel('x')
plt.ylabel('y')
plt.title('Solution y(x)')
plt.legend()

# Plot f(x) and its prediction
plt.subplot(1, 2, 2)
plt.plot(x_eval, f_x_true, label='True f(x) = 2x', color='blue')
plt.plot(x_eval, f_x_pred, label='Predicted f(x)', linestyle='--', color='red')
plt.xlabel('x')
plt.ylabel('f(x)')
plt.title('Function f(x)')
plt.legend()

plt.tight_layout()
plt.show()
