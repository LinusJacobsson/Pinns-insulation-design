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
from dataclasses import dataclass



@dataclass
class HarmonicConfig:
    m = 10
    mu = 5
    k = 10
    initial_x = 1
    initial_v = 0
    amplitude = 1
    train = True


# Hardcoded for initial_x = 1, initial_v = 0
def analytical_solution(t, m = HarmonicConfig.m, mu = HarmonicConfig.mu, k = HarmonicConfig.k):
    delta = mu/(2*m)
    omega_0 = np.sqrt(k/m)
    assert delta < omega_0
    omega = np.sqrt(omega_0**2 - delta**2)
    phi =  np.arctan(-delta/omega)
    A = 1/(2*np.cos(phi))
    cos = np.cos(phi + omega*t)
    exp = np.exp(-delta*t)
    return 2*A*exp*cos


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
    eq_loss = jnp.mean((d2x_dt2 + omega**2 * pred_displacement[:, 0] + 1)**2)

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


omega = 1
initial_x = -2.0
initial_v = 1.0

t = np.linspace(0, 10, 100)
y = analytical_solution(t)
t_samples = t[::5]
y_samples = y[::5]

plt.figure()
plt.plot(t, y, label="Exact solution")
plt.scatter(t_samples, y_samples, color="tab:orange", label="Training data")
plt.legend()
plt.show()

def main():
    for epoch in range(1000):
        state, loss = train_step(state, t_samples, omega, initial_x, initial_v)  # Removed 'model' from here
        if epoch % 100 == 0:
            print(f"Epoch {epoch}, Loss: {loss}")


    predict_fn = jax.jit(lambda t: model.apply(state.params, t))
    predicted_displacement = vmap(predict_fn)(t_samples)

    plt.figure(figsize=(10, 4))

    # Plotting the predicted displacement
    plt.plot(t, predicted_displacement, label='Predicted Displacement by PINN')

    # Plotting the true solution
    plt.plot(t, analytical_solution(t), label='True Solution', linestyle='dashed')

    # Adding collocation points on the plot
    # Assume `predicted_displacement_at_collocation` is the predicted displacement at collocation points
    #predicted_displacement_at_collocation = vmap(predict_fn)(collocation)
    #plt.scatter(collocation, predicted_displacement_at_collocation, color='red', s=10, label='Collocation Points')

    plt.xlabel('Time')
    plt.ylabel('Displacement')
    plt.title('Learned Simple Harmonic Oscillator vs True Solution')
    plt.legend()
    plt.grid(True)
    plt.show()

if HarmonicConfig.train:
    main()
