import jax.numpy as np
from jax import random, grad, vmap, jit
from jax.example_libraries import optimizers

import itertools
from tqdm import trange

import matplotlib.pyplot as plt

#%%
# Define MLP 
# activation function: tanh
# Initialization: Xavier Glorot 

def MLP(layers, activation=np.tanh):

    def init(rng_key):
        
      def init_layer(key, d_in, d_out):
          k1, k2 = random.split(key)
          glorot_stddev = 1.0 / np.sqrt((d_in + d_out) / 2.)
          W = glorot_stddev * random.normal(k1, (d_in, d_out))
          b = np.zeros(d_out)
          return W, b
      
      key, *keys = random.split(rng_key, len(layers))
      params = list(map(init_layer, keys, layers[:-1], layers[1:]))
      
      return params

    def apply(params, inputs):

        H = inputs
        for W, b in params[:-1]:
            outputs = np.dot(H, W) + b
            H = activation(outputs)
        W, b = params[-1]
        outputs = np.dot(H, W) + b
        
        return outputs
    
    return init, apply
     
#%% Damped vibration
def damped_vibration(m, k, c, x_0, v_0, t):
    
    wn = np.sqrt(k/m) # natural frequency
    c_cr = 2*m*wn # Critical damping
    zeta = c/c_cr # damping coefficient
    wd = np.sqrt(1-zeta**2)*wn # damped natuaral frequency
    
    # Initial condition constants
    X0 = (np.sqrt((x_0**2)*(wn**2)) + v_0**2 + 2*x_0*v_0*wn*zeta)/(wd)
    P0 = np.arctan((x_0*wd)/(v_0 + zeta*wn*x_0))
    
    # System response
    x = X0*np.exp(-zeta*wn*t)*np.sin(wd*t + P0)
    return x

# get the exact solution over the full domain
t_end = 2 # end time (s)

c = 2 # damping constant (kg/s)
m = 1 # mass (kg)
k = 100 # spring constant (N/m)
x_0 = 1 # initial displacement (m)
v_0 = 0 # initial velocity (m/s)

# Normalization over time domain
t = np.linspace(0,t_end,500)
scaler = 1/t.max()
t = t*scaler

c_scaled = c/(scaler) # normalized damping constant 
k_scaled = k/(scaler**2) # normalized spring constant
t_end_scaled = t_end*scaler

# Test points
t_ = np.linspace(0,t_end_scaled,500)
x_ = damped_vibration(m, k_scaled, c_scaled, x_0, v_0, t_)

# Training points
t = np.linspace(0,t_end_scaled,100) 
x = damped_vibration(m, k_scaled, c_scaled, x_0, v_0, t)

# Initial condition index
init_idt = np.isclose(t, t.min())

# Initial condition and collocation points
T_ic = t[init_idt]
X_ic = x[init_idt]
tr = t[~init_idt] # Collocation points

# Plot response and training points
plt.figure()
plt.plot(t_, x_, label="Exact solution")
plt.scatter(T_ic, X_ic, color="tab:orange", label="Initial condition data")
plt.scatter(tr, [0]*tr.shape[0], color="tab:green", label='collocation points', s=1)
plt.legend()
plt.show()

# Initalize the network
key = random.PRNGKey(12)
d0 = 1 # input dimension
layers = [d0, 128, 128, 1] # Network architecture
init, apply = MLP(layers, activation=np.tanh)
params = init(rng_key = key)

# Use optimizers to set optimizer initialization and update functions
lr = optimizers.exponential_decay(1e-3, decay_steps=5000, decay_rate=0.9)
opt_init, opt_update, get_params = optimizers.adam(lr)
opt_state = opt_init(params)

# Logger
itercount = itertools.count()
loss_log = []
loss_ics_log = []
loss_res_log = []
loss_vel_log = []
epoch_log = []
W_log = []
L_t_log = []

#%%

# Define network and ode functions
def neural_net(params, t):

    outputs = apply(params, t)
    
    return outputs[0][0]

def residual_net(params, t): 
    
    u = neural_net(params, t)
    u_x = grad(neural_net, argnums=1)(params, t)
    u_xx = grad(grad(neural_net, argnums = 1), argnums=1)(params, t)
    
    return m*u_xx + c_scaled*u_x + k_scaled*u 

def velocity_net(params, t):
    
    u_t = grad(neural_net, argnums=1)(params, t)
    
    return u_t

# vmap the functions
u_pred_fn = vmap(neural_net, (None, 0))
r_pred_fn = vmap(residual_net, (None, 0))
v_pred_fn = vmap(velocity_net, (None, 0))

#%% Training losses

# Initial displacement loss
@jit
def loss_ics(params):
    # Evaluate the network over IC
    u_pred = vmap(neural_net, (None, 0))(params, T_ic)
    # Compute the initial loss
    loss_ics = np.mean((X_ic.flatten() - u_pred.flatten())**2)
    return loss_ics

# Initial velocity loss
@jit
def loss_velocity(params):
    
    v_pred = v_pred_fn(params, T_ic)
    loss_v = np.mean(v_pred**2)
    
    return loss_v

# ode loss
@jit
def loss_res(params): 
    
    r_pred = r_pred_fn(params, tr)
    # Compute loss
    loss_r = np.mean(r_pred**2)
    
    return loss_r  

# Total loss
@jit
def loss(params):
    
    L0 = loss_ics(params) * 1000
    L_t = loss_res(params)
    L_v = loss_velocity(params)

    loss = L_t + L0 + L_v
    
    return loss

# Define a compiled update step
@jit
def step(i, opt_state):
    
    params = get_params(opt_state)
    g = grad(loss)(params)

    return opt_update(i, g, opt_state)

#%% Training

nIter = 40000 # number of epochs
pbar = trange(nIter)

# Main training loop
for it in pbar:
    current_count = next(itercount)
    opt_state = step(current_count, opt_state)
    
    if it % 200 == 0:
        params = get_params(opt_state)

        loss_value = loss(params)
        loss_ics_value = loss_ics(params)
        loss_res_value = loss_res(params)
        loss_v_value = loss_velocity(params)

        loss_log.append(loss_value)
        loss_ics_log.append(loss_ics_value)
        loss_res_log.append(loss_res_value)
        loss_vel_log.append(loss_v_value)
        epoch_log.append(current_count)
        

        pbar.set_postfix({'Loss': loss_value, 
                          'loss_ics' : loss_ics_value, 
                          'loss_res':  loss_res_value,
                          'loss_vel': loss_v_value})

#%% Plot training history and calculate relative l2 error

plt.plot(epoch_log, loss_ics_log)
plt.plot(epoch_log, loss_res_log)
plt.plot(epoch_log, loss_vel_log)
plt.yscale('log')

params = get_params(opt_state)
u_pred = u_pred_fn(params, t_)
error = np.linalg.norm(u_pred - x_) / np.linalg.norm(x_) 
print('Relative l2 error: {:.3e}'.format(error))
     
#%% Compare PINN and exact solutions
from matplotlib.pyplot import figure

figure(figsize=(8, 5))
plt.plot(t_/scaler, x_, label = 'Exact solution', c = '#e74f4d', linewidth=3)
plt.plot(t_/scaler, u_pred, '--', label = 'Prediction', c = '#474646', linewidth=3)

plt.grid(lw=0.5, c='darkgray', alpha=0.3)
plt.axhline(0,color='grey', alpha=0.4)
plt.axvline(0,color='grey', alpha=0.4)
plt.xticks(size=13, color='dimgray')
plt.yticks(size=13, color='dimgray')
plt.xlabel('Time', size=18, color='dimgray')
plt.ylabel('Displacement', size=18, color='dimgray')
plt.legend()

