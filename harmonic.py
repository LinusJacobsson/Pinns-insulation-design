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
from jax import debug


@dataclass
class HarmonicConfig:
    m = 1
    mu = 1
    k = 1
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


def model_loss(params, apply_fn, t, y, omega, initial_displacement, initial_velocity):
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
    d2x_dt2 = vmap(second_derivative)(t[:, 0])

    # Differential equation loss
    pred_displacement = displacement(t)
    eq_loss = jnp.mean((d2x_dt2 + omega**2 * pred_displacement[:, 0])**2)

    # Initial conditions loss
    ic_loss_displacement = (displacement(jnp.array([[0.]]))[0, 0] - initial_displacement) ** 2
    ic_loss_velocity = (grad(lambda t: displacement(jnp.array([[t]]))[0, 0])(0.0) - initial_velocity) ** 2

    data_loss = jnp.mean((pred_displacement - y)**2)
    
    total_loss = eq_loss + ic_loss_displacement + ic_loss_velocity + data_loss
    return total_loss, eq_loss, ic_loss_displacement, ic_loss_velocity, data_loss


@jax.jit
def train_step(state, t, y, omega, initial_displacement, initial_velocity):
    def loss_fn(params):
        total_loss, _, _, _, _ = model_loss(params, state.apply_fn, t, y, omega, initial_displacement, initial_velocity)
        return total_loss

    grad_fn = jax.value_and_grad(loss_fn)
    loss, grads = grad_fn(state.params)
    state = state.apply_gradients(grads=grads)
    return state, loss

def main():
        # Initialize model and optimizer
    model = PINN()
    rng = random.PRNGKey(0)
    params = model.init(rng, jnp.array([[0.]]))
    tx = optax.adam(learning_rate=0.001)
    state = train_state.TrainState.create(apply_fn=model.apply, params=params, tx=tx)


    omega = 1
    initial_x = 1.0
    initial_v = 0

    t = np.linspace(0, 20, 1000).reshape(-1, 1)
    y = analytical_solution(t)
    t_samples = np.concatenate([t[0:200:2], t[800:1000:2]])
    y_samples = np.concatenate([y[0:200:2], y[800:1000:2]])

    plt.figure()
    plt.plot(t, y, label="Exact solution")
    plt.scatter(t_samples, y_samples, color="tab:orange", label="Training data")
    plt.legend()
    plt.show()
    t_samples.reshape(-1, 1)

    for epoch in range(10000):
        state, _ = train_step(state, t_samples, y_samples, omega, initial_x, initial_v)
        if epoch % 1000 == 0:
            _, eq_loss, ic_loss_disp, ic_loss_vel, data_loss = model_loss(state.params, model.apply, t_samples, y_samples, omega, initial_x, initial_v)
            print(f"Epoch {epoch}, Equation Loss: {eq_loss}, IC Loss Displacement: {ic_loss_disp}, IC Loss Velocity: {ic_loss_vel}, Data Loss: {data_loss}")



    predict_fn = jax.jit(lambda t: model.apply(state.params, t))
    predicted_displacement = vmap(predict_fn)(t)
    plt.figure(figsize=(10, 4))
    # Plotting the predicted displacement
    plt.plot(t[:,], predicted_displacement, label='Predicted Displacement by PINN')
    # Plotting the true solution
    plt.plot(t[:, ], analytical_solution(t), label='True Solution', linestyle='dashed')
    plt.scatter(t_samples[:,], y_samples[:,], color = 'r', marker = '*')
    plt.xlabel('Time')
    plt.ylabel('Displacement')
    plt.title('Learned Simple Harmonic Oscillator vs True Solution')
    plt.legend()
    plt.grid(True)
    plt.show()

if HarmonicConfig.train:
    main()
