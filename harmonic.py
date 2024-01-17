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
