import jax 
import jax.numpy as jnp
import numpy as np
import optax
import flax.linen as nn

from typing import Sequence, Callable

import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib import cm

import numpy as np

def analytical_solution(t, x_0, v_0, m, mu, k):
    """
    Analytical solution for the damped harmonic oscillator with initial conditions.

    Parameters:
    - t (float or array-like): Time or array of times at which to evaluate the solution.
    - x_0 (float): Initial displacement.
    - v_0 (float): Initial velocity.
    - m (float): Mass of the oscillator.
    - mu (float): Damping coefficient.
    - k (float): Spring constant.

    Returns:
    - x (float or array-like): Displacement as a function of time.
    """
    omega_n = np.sqrt(k / m)  # Natural frequency
    zeta = mu / (2 * np.sqrt(m * k))  # Damping ratio
    omega_d = omega_n * np.sqrt(1 - zeta**2)  # Damped natural frequency

    # Underdamped solution
    e_term = np.exp(-zeta * omega_n * t)
    A = x_0
    B = (v_0 + zeta * omega_n * x_0) / omega_d

    x = e_term * (A * np.cos(omega_d * t) + B * np.sin(omega_d * t))
    return x



# N = 200 funkar ej, N = 100 gör det bra
def generate_dataset(N=50, noise_percent=0.0, omega=8, seed=420, x_0 = -2, v_0 = 0, m = 1, mu = 1, k = 1):
    # seed key for debugging
    np.random.seed(seed)

    # Domains of t and x
    tmin, tmax = 0.0, 5.0

    t_vals = np.random.uniform(low=tmin,high=tmax,size=(N,1))

    u_vals = analytical_solution(t=t_vals, x_0=x_0, v_0=v_0, m=m, mu=mu, k=k)
    noise = np.random.normal(0, u_vals.std(), [N,1])*noise_percent
    u_vals += noise

    colloc = jnp.concatenate([t_vals, u_vals],axis=1)

    return colloc, tmin, tmax

data, tmin, tmax = generate_dataset()


class MLP(nn.Module):
    features: Sequence[int]
    # We also add an initializer for the omega parameter
    m_init: Callable = jax.nn.initializers.ones
    mu_init: Callable = jax.nn.initializers.ones
    k_init: Callable = jax.nn.initializers.ones


    def setup(self):
        # include the omega parameter during setup
        m = self.param("m", self.m_init, (1,))
        mu = self.param("mu", self.mu_init, (1,))
        k = self.param("k", self.k_init, (1,))

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
    
def PINN_f(t, m, mu, k, ufunc):
    # First derivative of ufunc (velocity)
    u_t = jax.grad(lambda t: jnp.sum(ufunc(t)))(t)

    # Second derivative of ufunc (acceleration)
    u_tt = jax.grad(lambda t: jnp.sum(u_t(t)))(t)

    # Damped harmonic oscillator equation: m*u_tt + mu*u_t + k*ufunc = 0
    # Rearranged to form: m*u_tt + mu*u_t + k*ufunc
    return m * u_tt + mu * u_t + k * ufunc(t)


@jax.jit
def uNN(params, t):
    u = model.apply(params, t)
    return u

def loss_fun(params,data):
    t_c, u_c = data[:,[0]], data[:,[1]]
    ufunc = lambda t : uNN(params,t)
   
    m = params["params"]["m"]
    mu = params["params"]["mu"]
    k = params["params"]["k"]

    mse_u = MSE(u_c, ufunc(t_c))
    mse_f = jnp.mean(PINN_f(t_c, m, mu, k, ufunc)**2)
    
    return mse_f + mse_u

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

    lr = optax.piecewise_constant_schedule(1e-2,{50_000:5e-3,80_000:1e-3})
    optimizer = optax.adam(lr)
    opt_state = optimizer.init(params)
    
    return model, params, optimizer, opt_state

features = [8,  8, 1]

model, params, optimizer, opt_state = init_process(features)


epochs = 10_000
for epoch in range(epochs):
    opt_state, params = update(opt_state,params,data)
    current_m = params["params"]["m"][0]
    current_mu = params["params"]["mu"][0]
    current_k = params["params"]["k"][0]
    # print loss and epoch info
    if epoch%(1000) ==0:
        print(f'Epoch = {epoch},\tloss = {loss_fun(params,data):.3e},\tm = {current_m:.3f},\tmu = {current_mu:.3f},\tk = {current_k:.3f}')

Dcalc = params["params"]["omega"][0]
omega = 8
print(f"The real value of the parameter is omega = {omega}")
print(f"The calculated value for the parameter is D_calc = {Dcalc:.7f}.")
print(f"This corresponds to a {100*(Dcalc-omega)/omega:.5f}% error.")



# Generate a set of time points for evaluation
t_eval = np.linspace(tmin, tmax, 500)[:, None]

# Compute the analytical solution
analytical_solution = analytic(t=t_eval, x_0=-2, v_0=0, omega=8)

# Compute the neural network prediction
nn_solution = uNN(params, jnp.array(t_eval))

# Plotting
plt.figure(figsize=(10, 6))
plt.plot(t_eval, analytical_solution, label='Analytical Solution', color='blue')
plt.plot(t_eval, nn_solution, label='NN Prediction', linestyle='--', color='red')
plt.xlabel('Time')
plt.ylabel('Displacement')
plt.title('Comparison of Analytical and NN Solutions for the SHO')
plt.legend()
plt.show()