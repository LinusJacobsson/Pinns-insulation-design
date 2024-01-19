import jax 
import jax.numpy as jnp
import numpy as np
import optax
import flax.linen as nn

from typing import Sequence, Callable

import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib import cm

def analytic_u(t,x,D):
    return np.sin(np.pi*x)*np.exp(-D*(np.pi**2)*t)


def plot_initial(data):
    
    fig = plt.figure(figsize=(6,4))

    cmap = 'Spectral'
    
    t, x, u = data[:,[0]], data[:,[1]], data[:,[2]]

    plt.scatter(t, x, c=u, marker='x', vmin=0, vmax=1, cmap=cmap)

    plt.xlabel(r'$t$')
    plt.ylabel(r'$x$')

    plt.title(f'Training Dataset, N = {len(t)}')

    cbar = plt.colorbar(mpl.cm.ScalarMappable(cmap=cmap))
    cbar.set_label(r'$u(t,x)$', rotation=90)

    plt.show()

def generate_dataset(N=1000, noise_percent=0.0, D=0.1, seed=420):
    # seed key for debugging
    np.random.seed(seed)

    # Domains of t and x
    tmin, tmax = 0.0, 1.0
    xmin, xmax = 0.0, 1.0

    t_vals = np.random.uniform(low=tmin,high=tmax,size=(N,1))
    x_vals = np.random.uniform(low=xmin,high=xmax,size=(N,1))

    u_vals = analytic_u(t=t_vals,x=x_vals,D=D)
    noise = np.random.normal(0, u_vals.std(), [N,1])*noise_percent
    u_vals += noise

    colloc = jnp.concatenate([t_vals,x_vals,u_vals],axis=1)

    return colloc, tmin, tmax, xmin, xmax

data, tmin, tmax, xmin, xmax = generate_dataset()


class MLP(nn.Module):
    features: Sequence[int]
    # We also add an initializer for the D parameter
    D_init: Callable = jax.nn.initializers.ones

    def setup(self):
        # include the D parameter during setup
        D = self.param("D", self.D_init, (1,))
        self.layers = [nn.Dense(features=feat, use_bias=True) for feat in self.features]
        
    def __call__(self, inputs):
        x = inputs
        for idx, layer in enumerate(self.layers):
            x = layer(x)
            if idx != len(self.layers)-1:
                x = jnp.tanh(x)
        return x
    

@jax.jit
def MSE(true,pred):
    return jnp.mean((true-pred)**2)
    
def PINN_f(t,x,D,ufunc):
    u_x = lambda t,x : jax.grad(lambda t,x : jnp.sum(ufunc(t,x)),1)(t,x)
    u_xx = lambda t,x : jax.grad(lambda t,x : jnp.sum(u_x(t,x)),1)(t,x)
    u_t = lambda t,x : jax.grad(lambda t,x : jnp.sum(ufunc(t,x)),0)(t,x)
    return u_t(t,x) - D*u_xx(t,x)
    
@jax.jit
def uNN(params,t,x):
    u = model.apply(params, jnp.concatenate((t,x),axis=1))
    return u

def loss_fun(params,data):
    t_c, x_c, u_c = data[:,[0]], data[:,[1]], data[:,[2]]
    ufunc = lambda t,x : uNN(params,t,x)
    
    # Find the value of D
    D = params["params"]["D"]
    
    mse_u = MSE(u_c,ufunc(t_c,x_c))
    mse_f = jnp.mean(PINN_f(t_c,x_c,D,ufunc)**2)
    
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

    dummy_in = jax.random.normal(key1, (2,))
    params = model.init(key2, dummy_in)

    lr = optax.piecewise_constant_schedule(1e-2,{10_000:5e-3,30_000:1e-3,50_000:5e-4,70_000:1e-4})
    optimizer = optax.adam(lr)
    opt_state = optimizer.init(params)
    
    return model, params, optimizer, opt_state

def plot_results(T, X, Dcalc, results, analytic_results):
    fig, axes = plt.subplots(2, 1, figsize=(4,8))

    t_ticks = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]
    x_ticks = [0.0, 0.25, 0.5, 0.75, 1.0]

    c1 = axes[0].pcolormesh(T, X, results, cmap='Spectral')
    axes[0].set(xlabel = r'$t$', ylabel = r'$x$', xticks=t_ticks, yticks=x_ticks)
    axes[0].set_title(f'PINN Solution of the PDE, with D = {Dcalc:.7f}')

    cbar1 = plt.colorbar(c1, ax=axes[0])
    cbar1.set_label(r'$u(t,x)$', rotation=90)

    c2 = axes[1].pcolormesh(T, X, abs(analytic_results-results), cmap='Spectral')
    axes[1].set(xlabel = r'$t$', ylabel = r'$x$', xticks=t_ticks, yticks=x_ticks)
    axes[1].set_title('Absolute difference between PINN and exact solution')

    cbar2 = plt.colorbar(c2, ax=axes[1])
    cbar2.set_label(r'$|{u_{exact}}(t,x) - u(t,x)|$', rotation=90)

    plt.show()

features = [8, 8, 8, 8, 8, 8, 1]

model, params, optimizer, opt_state = init_process(features)


epochs = 10_000
for epoch in range(epochs):
    opt_state, params = update(opt_state,params,data)

    # print loss and epoch info
    if epoch%(1000) ==0:
        print(f'Epoch = {epoch},\tloss = {loss_fun(params,data):.3e}')

Dcalc = params["params"]["D"][0]
D = 0.1
print(f"The real value of the parameter is D = {D}")
print(f"The calculated value for the parameter is D_calc = {Dcalc:.7f}.")
print(f"This corresponds to a {100*(Dcalc-D)/D:.5f}% error.")


N_grid = 200 # Defines the grid on which to draw the solution
tspace = np.linspace(tmin, tmax, N_grid)
xspace = np.linspace(xmin, xmax, N_grid)
T, X = np.meshgrid(tspace, xspace)

results = uNN(params,T.flatten().reshape(-1,1),X.flatten().reshape(-1,1)).reshape(N_grid,N_grid)
analytic_results = analytic_u(T.flatten().reshape(-1,1),X.flatten().reshape(-1,1),D).reshape(N_grid,N_grid)

plot_results(T, X, Dcalc, results, analytic_results)