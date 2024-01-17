# First implementation of PINN on a simple harmonic oscillator using FLAX
import jax
import jax.numpy as jnp
from jax import random, grad, vmap
import flax
import flax.linen as nn
from flax.training import train_state
import optax
import matplotlib.pyplot as plt
import numpy as np

class PINN(nn.Module):
    @nn.compact
    def __call__(self, t):
        x = nn.Dense(200)(t)
        x = nn.tanh(x)
        x = nn.Dense(200)(x)
        x = nn.tanh(x)
        x = nn.Dense(1)(x)
        return x


def model_loss(params, apply_fn, inputs, omega, initial_displacement, initial_velocity):
    def displacement(t):
        t_reshaped = t.reshape(-1, 1)  # Reshape to (batch_size, num_features)
        return apply_fn(params, t_reshaped)

    # Compute the second derivative for each input point individually
    def second_derivative(t):
        # First derivative
        dx_dt = grad(lambda t: displacement(jnp.array([t]))[0, 0])(t)
        # Second derivative
        return grad(lambda t: dx_dt)(t)

    # Vectorize the second derivative computation
    d2x_dt2 = vmap(second_derivative)(inputs[:, 0])

    # Differential equation loss
    pred_displacement = displacement(inputs)
    eq_loss = jnp.mean((d2x_dt2 + omega**2 * pred_displacement[:, 0])**2)

    # Initial conditions loss
    ic_loss_displacement = (displacement(jnp.array([[0.]]))[0, 0] - initial_displacement) ** 2
    ic_loss_velocity = (grad(lambda t: displacement(jnp.array([[t]]))[0, 0])(0.0) - initial_velocity) ** 2


    return eq_loss + ic_loss_displacement + ic_loss_velocity


@jax.jit
def train_step(state, inputs, omega, initial_displacement, initial_velocity):
    def loss_fn(params):
        # Use the apply function from the state object
        return model_loss(params, state.apply_fn, inputs, omega, initial_displacement, initial_velocity)

    grad_fn = jax.value_and_grad(loss_fn)
    loss, grads = grad_fn(state.params)
    state = state.apply_gradients(grads=grads)
    return state, loss


# Initialize model and optimizer
model = PINN()
rng = random.PRNGKey(0)
params = model.init(rng, jnp.array([[0.]]))
tx = optax.adam(learning_rate=0.001)
state = train_state.TrainState.create(apply_fn=model.apply, params=params, tx=tx)

# Training data
times = jnp.linspace(0, 10, 1000).reshape(-1, 1)
collocation = times[::30]

omega = 1.0
initial_x = -2.0
initial_v = 0.0

# Training loop

for epoch in range(10000):
    state, loss = train_step(state, collocation, omega, initial_x, initial_v)  # Removed 'model' from here
    if epoch % 1000 == 0:
        print(f"Epoch {epoch}, Loss: {loss}")



predict_fn = jax.jit(lambda t: model.apply(state.params, t))
predicted_displacement = vmap(predict_fn)(collocation)

true_solution = initial_x * np.cos(omega * times)

plt.figure(figsize=(10, 4))

# Plotting the predicted displacement
plt.plot(collocation, predicted_displacement, label='Predicted Displacement by PINN')

# Plotting the true solution
plt.plot(times, true_solution, label='True Solution', linestyle='dashed')

# Adding collocation points on the plot
# Assume `predicted_displacement_at_collocation` is the predicted displacement at collocation points
predicted_displacement_at_collocation = vmap(predict_fn)(collocation)
plt.scatter(collocation, predicted_displacement_at_collocation, color='red', s=10, label='Collocation Points')

plt.xlabel('Time')
plt.ylabel('Displacement')
plt.title('Learned Simple Harmonic Oscillator vs True Solution')
plt.legend('top right')
plt.grid(True)
plt.show()

