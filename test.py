import jax 
import jax.numpy as jnp
import numpy as np
import optax
import flax.linen as nn

from typing import Sequence, Callable

import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib import cm

def analytic(t, x_0, v_0, omega):
    """
    Analytical solution for the simple harmonic oscillator with initial conditions.

    Parameters:
    - t (float or array-like): Time or array of times at which to evaluate the solution.
    - x_0 (float): Initial displacement.
    - v_0 (float): Initial velocity.
    - omega (float): Angular frequency.

    Returns:
    - x (float or array-like): Displacement as a function of time.
    """
    A = x_0  # Amplitude from initial displacement
    B = v_0 / omega  # Amplitude from initial velocity
    x = A * np.cos(omega * t) + B * np.sin(omega * t)
    return x
# N = 200 funkar ej, N = 100 gör det bra
def generate_dataset(N=1000, noise_percent=0.0, omega=4, seed=420, x_0 = -2, v_0 = 0):
    # seed key for debugging
    np.random.seed(seed)

    # Domains of t and x
    tmin, tmax = 0.0, 10.0

    t_vals = np.random.uniform(low=tmin,high=tmax,size=(N,1))

    u_vals = analytic(t=t_vals,omega=omega, x_0=x_0, v_0=v_0)
    noise = np.random.normal(0, u_vals.std(), [N,1])*noise_percent
    u_vals += noise

    colloc = jnp.concatenate([t_vals, u_vals],axis=1)

    return colloc, tmin, tmax

data, tmin, tmax = generate_dataset()


class MLP(nn.Module):
    features: Sequence[int]
    # We also add an initializer for the omega parameter
    omega_init: Callable = jax.nn.initializers.ones

    def setup(self):
        # include the omega parameter during setup
        omega = self.param("omega", self.omega_init, (1,))
        self.layers = [nn.Dense(features=feat, use_bias=True) for feat in self.features]
        
    def __call__(self, inputs):
        x = inputs
        for idx, layer in enumerate(self.layers):
            x = layer(x)
            if idx != len(self.layers)-1:
                x = jnp.tanh(x)
        return x
    

@jax.jit
def MSE(true, pred):
    return jnp.mean((true-pred)**2)
    
def PINN_f(t, omega, ufunc):
    u_t = lambda t: jax.grad(lambda t: jnp.sum(ufunc(t)))(t)
    u_tt = lambda t: jax.grad(lambda t: jnp.sum(u_t(t)))(t)
    return u_tt(t) + omega**2 * ufunc(t)  # Corrected to omega**2 as per the SHO equation
    

@jax.jit
def uNN(params, t):
    u = model.apply(params, t)
    return u

def loss_fun(params,data):
    t_c, u_c = data[:,[0]], data[:,[1]]
    ufunc = lambda t : uNN(params,t)
    
    # Find the value of D
    omega = params["params"]["omega"]
    
    mse_u = MSE(u_c, ufunc(t_c))
    mse_f = jnp.mean(PINN_f(t_c,omega,ufunc)**2)
    
    return mse_f + 10*mse_u

@jax.jit
def update(opt_state,params,data):
    # Get the gradient w.r.t to MLP params
    grads=jax.jit(jax.grad(loss_fun,0))(params, data)

    # Update params
    updates, opt_state = optimizer.update(grads, opt_state)
    
    # Apply the update
    params = optax.apply_updates(params, updates)

    return opt_state, params

def init_process(feats):
    
    model = MLP(features=feats)

    key1, key2 = jax.random.split(jax.random.PRNGKey(420),num=2)

    dummy_in = jax.random.normal(key1, (1,))
    params = model.init(key2, dummy_in)

    lr = optax.piecewise_constant_schedule(1e-2,{15_000:5e-3,80_000:1e-3})
    optimizer = optax.adam(lr)
    opt_state = optimizer.init(params)
    
    return model, params, optimizer, opt_state

features = [8, 8, 8, 8, 8, 1]

model, params, optimizer, opt_state = init_process(features)


epochs = 100_000
for epoch in range(epochs):
    opt_state, params = update(opt_state,params,data)
    current_omega = params["params"]["omega"][0]
    # print loss and epoch info
    if epoch%(1000) ==0:
        print(f'Epoch = {epoch},\tloss = {loss_fun(params,data):.3e}, \t omega = {current_omega:.3f}')

Dcalc = params["params"]["omega"][0]
omega = 4
print(f"The real value of the parameter is omega = {omega}")
print(f"The c