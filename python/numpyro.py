#%%  
import numpyro as npy
import numpy as np
import jax.numpy as jnp
import jax
from jax.experimental.ode import odeint
import jax.numpy as jnp
from jax.random import PRNGKey
import matplotlib.pyplot as plt
#%% 
nL = np.genfromtxt('../graphs/ER-10-05.csv', delimiter=',')
# %%
L = jnp.array(nL)
# %%
def NetworkFKPP(u, t, p):
    k, a, L = p
    dudt = -k * L @ u + a * u * (1 - u)
    return dudt

# %%
u = np.zeros(10)
u[5] = 0.1
u0 = jnp.array(u)
# %%
p = [1.0,2.0,L]
# %%
t = jnp.arange(0,10,0.1)
# %%
sol = odeint(NetworkFKPP, u0, t, p)
# %%
plt.plot(t, sol)
plt.show()
# %%
