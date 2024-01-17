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


