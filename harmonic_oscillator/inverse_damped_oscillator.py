import jax 
import jax.numpy as jnp
import numpy as np
import optax
import flax.linen as nn
from typing import Sequence, Callable
import matplotlib.pyplot as plt


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
def generate_dataset(N=100, noise_percent= 0.5, seed=420, x_0 = -2, v_0 = 0, m = 1, mu = 1.0, k = 4.0):
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
    # Modify initializers to use different distributions or values
    #m_init: Callable = lambda key, shape: jax.random.uniform(key, shape, minval=1.0, maxval=10.0)
    # Custom uniform initializer for mu
    #mu_init: Callable = lambda key, shape: jax.random.uniform(key, shape, minval=1.0, maxval=10.0)
    # Constant value initializer for k
    m_init: Callable = jax.nn.initializers.ones
    mu_init: Callable = jax.nn.initializers.ones
    k_init: Callable = jax.nn.initializers.ones

    def setup(self):
        # include the omega parameter during setup
        #m = self.param("m", self.m_init, (1,))
        #mu = self.param("mu", self.mu_init, (1,))
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
    u_t = lambda t: jax.grad(lambda t: jnp.sum(ufunc(t)))(t)
    u_tt = lambda t: jax.grad(lambda t: jnp.sum(u_t(t)))(t)
    return m*u_tt(t) + (mu)*u_t(t) + (k)*ufunc(t) 

@jax.jit
def uNN(params, t):
    u = model.apply(params, t)
    return u

@jax.jit
def loss_fun(params, data, x_0, v_0):
    t_c, u_c = data[:, [0]], data[:, [1]]
    ufunc = lambda t: uNN(params, t)
   
    # Assuming m and mu are fixed and known, using 1 for simplicity
    m = params["params"]["m"]
    mu = params["params"]["mu"]
    k = params["params"]["k"]

    mse_u = MSE(u_c, ufunc(t_c))
    mse_f = jnp.mean(PINN_f(t_c, m, mu, k, ufunc)**2)

    total_loss = mse_f + 1000*mse_u
    # Combine losses
    return total_loss, mse_f, mse_u


@jax.jit
def update(opt_state, params, data, x_0, v_0):
    # Extract only the total loss for gradient calculation
    total_loss, _, _ = loss_fun(params, data, x_0, v_0)
    grads = jax.grad(lambda p: loss_fun(p, data, x_0, v_0)[0])(params)  # [0] to get the total loss
    updates, opt_state = optimizer.update(grads, opt_state)
    params = optax.apply_updates(params, updates)
    return opt_state, params


def init_process(feats):
    
    model = MLP(features=feats)

    key1, key2 = jax.random.split(jax.random.PRNGKey(420),num=2)

    dummy_in = jax.random.normal(key1, (1,))
    params = model.init(key2, dummy_in)

    lr = optax.piecewise_constant_schedule(1e-4,{15_000:5e-3,90_000:1e-3})
    optimizer = optax.adam(lr)
    opt_state = optimizer.init(params)
    
    return model, params, optimizer, opt_state

features = [64, 64, 1]

model, params, optimizer, opt_state = init_process(features)

t_c, u_c = data[:, [0]], data[:, [1]]
ufunc = lambda t: uNN(params, t)
x_0 = -2
v_0 = 0
epochs = 20000
for epoch in range(epochs):
    opt_state, params = update(opt_state, params, data, x_0, v_0)
    total_loss, mse_f_loss, mse_u_loss = loss_fun(params, data, x_0, v_0)
    
    if epoch % 1000 == 0:
        current_m = params["params"]["m"][0]
        current_mu = params["params"]["mu"][0]
        current_k = params["params"]["k"][0]
        print(f'Epoch = {epoch}, Total Loss = {total_loss:.3e}, MSE_F = {mse_f_loss:.3e}, '
              f'MSE_U = {mse_u_loss:.3e}, m  = {current_m:.3f}, mu = {current_mu:.3f}, k = {current_k:.3f}')


m_calc = params["params"]["m"][0]
mu_calc = params["params"]["mu"][0]
k_calc = params["params"]["k"][0]

m = 1
mu = 1
k = 4

"""print(f"The real value of the parameter is m = {m}")
print(f"The calculated value for the parameter is m_calc = {m_calc:.7f}.")
print(f"This corresponds to a {100*(m_calc-m)/m:.5f}% error.")

print(f"The real value of the parameter is mu/m = {mu}")
print(f"The calculated value for the parameter is mu_calc = {mu_calc:.7f}.")
print(f"This corresponds to a {100*(mu_calc-mu)/mu:.5f}% error.")

print(f"The real value of the parameter is k = {k}")
print(f"The calculated value for the parameter is k_calc = {k_calc:.7f}.")
print(f"This corresponds to a {100*(k_calc-k)/k:.5f}% error.")
"""
print(f"The real value of the quotient m/k = {m/k}")
print(f"The calculated value for the parameter is m_calc/k_calc = {m_calc/k_calc:.7f}.")
print(f"This corresponds to a {100*((m_calc/k_calc)-(m/k))/(m/k):.5f}% error.")

print(f"The real value of the parameter is mu/m = {mu/m}")
print(f"The calculated value for the parameter is mu_calc/m_calc = {mu_calc/m_calc:.7f}.")
print(f"This corresponds to a {100*((mu_calc/m_calc)-(mu/m))/(mu/m):.5f}% error.")


# Generate a set of time points for evaluation
t_eval = np.linspace(tmin, tmax, 500)[:, None]

# Compute the analytical solution
solution = analytical_solution(t=t_eval, x_0=-2, v_0=0, m=m, mu=mu, k=k)

# Compute the neural network prediction
nn_solution = uNN(params, jnp.array(t_eval))

# Plotting
plt.figure(figsize=(10, 6))
plt.plot(t_eval, solution, label='Analytical Solution', color='blue')
plt.scatter(data[:, 0], data[:, 1], color='green', label='Training Data')
plt.plot(t_eval, nn_solution, label='NN Prediction', linestyle='--', color='red')
plt.xlabel('Time')
plt.ylabel('Displacement')
plt.title('Comparison of Analytical and NN Solutions for the SHO')
plt.legend()
plt.show()