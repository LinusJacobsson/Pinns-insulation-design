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
from IPython.display import clear_output



@dataclass
class HarmonicConfig:
    m = 1.5
    mu = 0
    k = 1.5
    initial_x = -2
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
    num_inputs: int
    num_outputs: int
    num_hidden: int
    num_layers: int

    @nn.compact
    def __call__(self, x):
        # Input layer
        x = nn.Dense(self.num_hidden)(x)
        x = nn.tanh(x)
        
        # Hidden layers
        for _ in range(self.num_layers - 1):
            x = nn.Dense(self.num_hidden)(x)
            x = nn.tanh(x)
        
        # Output layer
        x = nn.Dense(self.num_outputs)(x)
        return x

def model_loss(params, apply_fn, t_samples, y_samples, t_physics, omega, initial_displacement, initial_velocity):
    def displacement(t):
        t_reshaped = t.reshape(-1, 1)
        return apply_fn(params, t_reshaped)
    
    def first_derivative(t):
        return grad(lambda t: displacement(jnp.array([t]))[0, 0])(t)

    def second_derivative(t):
        dx_dt = grad(lambda t: displacement(jnp.array([t]))[0, 0])(t)
        return grad(lambda t: dx_dt)(t)

    # Use t_physics for the equation loss
    dx_dt = vmap(first_derivative)(t_physics[:, 0])
    d2x_dt2_physics = vmap(second_derivative)(t_physics[:, 0])
    pred_displacement_physics = displacement(t_physics)
    eq_loss = jnp.mean((HarmonicConfig.m*d2x_dt2_physics/HarmonicConfig.k + (HarmonicConfig.mu/HarmonicConfig.k)*dx_dt + pred_displacement_physics[:, 0])**2)
    #eq_loss = 0
    # Use t_samples and y_samples for the data loss
    pred_displacement_data = displacement(t_samples)
    data_loss = jnp.mean((pred_displacement_data - y_samples)**2)

    # Initial conditions loss (can use t_samples[0] if it starts from t=0)
    ic_loss_displacement = (displacement(jnp.array([[0.]]))[0, 0] - initial_displacement) ** 2
    ic_loss_velocity = (grad(lambda t: displacement(jnp.array([[t]]))[0, 0])(0.0) - initial_velocity) ** 2

    total_loss = eq_loss + ic_loss_displacement + ic_loss_velocity + 10*data_loss
    return total_loss, eq_loss, ic_loss_displacement, ic_loss_velocity, data_loss

def pde_residual_gradient(t, apply_fn, params):
    # Gradient of the PDE residual
    grad_pde_residual = grad(lambda t: pde_residual(t, apply_fn, params))
    # Vectorize the gradient computation over t
    return vmap(grad_pde_residual)(t)


def pde_residual(t, apply_fn, params):
    # Reshape t to a 2D array for displacement computation
    t_reshaped = jnp.array([t])
    displacement_val = apply_fn(params, t_reshaped)[0, 0]
    d2x_dt2 = second_derivative(t[0], apply_fn, params)
    residual = HarmonicConfig.m * d2x_dt2 + HarmonicConfig.k * displacement_val
    return residual

def second_derivative(t, apply_fn, params):
    def displacement_scalar(t_scalar):
        t_array = jnp.array([[t_scalar]])
        return apply_fn(params, t_array)[0, 0]

    dx_dt = grad(displacement_scalar)(t)
    d2x_dt2 = grad(lambda t: dx_dt)(t)
    return d2x_dt2



@jax.jit
def train_step(state, t_samples, y_samples, t_physics, omega, initial_displacement, initial_velocity):
    def loss_fn(params):
        total_loss, _, _, _, _ = model_loss(params, state.apply_fn, t_samples, y_samples, t_physics, omega, initial_displacement, initial_velocity)
        return total_loss

    grad_fn = jax.value_and_grad(loss_fn)
    loss, grads = grad_fn(state.params)
    state = state.apply_gradients(grads=grads)
    return state, loss

def main():
        # Initialize model and optimizer
    model = PINN(num_inputs=1, num_outputs=1, num_hidden=8, num_layers=3)
    rng = random.PRNGKey(0)
    params = model.init(rng, jnp.array([[0.]]))
    tx = optax.adam(learning_rate=0.001)
    state = train_state.TrainState.create(apply_fn=model.apply, params=params, tx=tx)


    omega = 1
    initial_x = -2.0
    initial_v = 0

    t = np.linspace(0, 8, 100).reshape(-1, 1)
    y = analytical_solution(t)
    #t_samples = np.concatenate([t[0:200:2], t[800:1000:2]])
    #y_samples = np.concatenate([y[0:200:2], y[800:1000:2]])
    t_samples = t[0:50:5]
    y_samples = y[0:50:5]
    t_physics = np.linspace(0, 8, 20).reshape(-1, 1)



    plt.figure()
    plt.plot(t, y, label="Exact solution")
    plt.scatter(t_samples, y_samples, color="tab:orange", label="Training data")
    plt.legend()
    plt.show()
    t_samples.reshape(-1, 1)

    for epoch in range(10000):
        state, _ = train_step(state, t_samples, y_samples, t_physics, omega, initial_x, initial_v)
        if epoch % 1000 == 0:
            _, eq_loss, ic_loss_disp, ic_loss_vel, data_loss = model_loss(state.params, model.apply, t_samples, y_samples, t_physics, omega, initial_x, initial_v)
            print(f"Epoch {epoch:.2f}, Equation Loss: {eq_loss:.2f}, IC Loss Displacement: {ic_loss_disp:.2f}, IC Loss Velocity: {ic_loss_vel:.2f}, Data Loss: {data_loss:.2f}")

    pde_gradients = pde_residual_gradient(t_physics, model.apply, state.params)

    predict_fn = jax.jit(lambda t: model.apply(state.params, t))
    predicted_displacement = vmap(predict_fn)(t)
    

    # Plotting results
    fig, ax1 = plt.subplots()

    # Displacement plot
    ax1.set_xlabel('Time')
    ax1.set_ylabel('Displacement')
    ax1.plot(t[:, 0], predicted_displacement, label='Predicted Displacement by PINN')
    ax1.plot(t[:, 0], analytical_solution(t), label='True Solution', linestyle='dashed')
    ax1.scatter(t_samples[:, 0], y_samples[:, 0], color='r', marker='o', label='Data Points')
    ax1.legend()
    ax1.set_ylim(-4, 4)
    ax1.grid(True)

    # Create a twin axis for the gradient plot
    ax2 = ax1.twinx()
    ax2.set_ylabel('PDE Residual Gradient', color='tab:purple')
    ax2.plot(t_physics[:, 0], pde_gradients, color='tab:purple', label='PDE Residual Gradient', linestyle=':')
    ax2.tick_params(axis='y', labelcolor='tab:purple')

    fig.tight_layout()
    plt.title('Learned Simple Harmonic Oscillator and PDE Residual Gradient')
    plt.show()


if HarmonicConfig.train:
    main()
