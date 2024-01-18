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
    m = 1
    mu = 4
    k = 400
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
    num_inputs: int
    num_outputs: int
    num_hidden: int
    num_layers: int
    m: float  # initial guess for mass
    mu: float  # initial guesss for damping coefficient
    k: float  # initial guess for spring constant

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

        # Trainable physical parameters
        m = self.param('m', nn.initializers.constant(self.m), ())
        mu = self.param('mu', nn.initializers.constant(self.mu), ())
        k = self.param('k', nn.initializers.constant(self.k), ())

        return x, m, mu, k


def model_loss(params, apply_fn, t_samples, y_samples, t_physics, omega, initial_displacement, initial_velocity):

    # Unpack trainable parameters from the model
    _, m, mu, k = apply_fn(params, jnp.zeros((1, 1)))

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
    eq_loss = jnp.mean((m*d2x_dt2_physics + mu*dx_dt + k*pred_displacement_physics[:, 0])**2)
    #eq_loss = 0
    # Use t_samples and y_samples for the data loss
    pred_displacement_data = displacement(t_samples)
    data_loss = jnp.mean((pred_displacement_data - y_samples)**2)

    # Initial conditions loss (can use t_samples[0] if it starts from t=0)
    ic_loss_displacement = (displacement(jnp.array([[0.]]))[0, 0] - initial_displacement) ** 2
    ic_loss_velocity = (grad(lambda t: displacement(jnp.array([[t]]))[0, 0])(0.0) - initial_velocity) ** 2

    total_loss = eq_loss + ic_loss_displacement + ic_loss_velocity + 10*data_loss
    return total_loss, eq_loss, ic_loss_displacement, ic_loss_velocity, data_loss



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
    model = PINN(num_inputs=1, num_outputs=1, num_hidden=8, num_layers=3, m=10.0, mu=10.0, k=100.0)
    rng = random.PRNGKey(0)
    params = model.init(rng, jnp.array([[0.]]))
    tx = optax.adam(learning_rate=0.001)
    state = train_state.TrainState.create(apply_fn=model.apply, params=params, tx=tx)


    omega = 20
    initial_x = 1.0
    initial_v = 0

    t = np.linspace(0, 1, 100).reshape(-1, 1)
    y = analytical_solution(t)
    #t_samples = np.concatenate([t[0:200:2], t[800:1000:2]])
    #y_samples = np.concatenate([y[0:200:2], y[800:1000:2]])
    t_samples = t[0:50:5]
    y_samples = y[0:50:5]
    t_physics = np.linspace(0, 1, 20).reshape(-1, 1)



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
            print(f"Epoch {epoch}, Equation Loss: {eq_loss}, IC Loss Displacement: {ic_loss_disp}, IC Loss Velocity: {ic_loss_vel}, Data Loss: {data_loss}")
            # Unpack and print the physical parameters
            _, m, mu, k = model.apply(state.params, jnp.zeros((1, 1)))
            print(f"Epoch {epoch}, m: {m}, mu: {mu}, k: {k}, Loss: {_}")

    predict_fn = jax.jit(lambda t: model.apply(state.params, t))
    predicted_displacement = vmap(predict_fn)(t)
    

    # Plotting the predicted displacement
    plt.plot(t[:, 0], predicted_displacement, label='Predicted Displacement by PINN')
    
    # Plotting the true solution
    plt.plot(t[:, 0], analytical_solution(t), label='True Solution', linestyle='dashed')
    
    # Plotting the data loss points
    plt.scatter(t_samples[:, 0], y_samples[:, 0], color='r', marker='o', label='Data Loss Points')

    # Plotting the physics loss points at y=0 (on the x-axis)
    plt.scatter(t_physics[:, 0], np.zeros_like(t_physics[:, 0]), color='b', marker='.', label='Physics Loss Points')

    plt.xlabel('Time')
    plt.ylabel('Displacement')
    plt.title('Learned Simple Harmonic Oscillator vs True Solution')
    plt.legend()
    plt.grid(True)
    plt.show()


if HarmonicConfig.train:
    main()
