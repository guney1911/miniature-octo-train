from numba import jit
import numpy as np
"""
This file contains numba versions of the functions used by the solver. 
Since numba does not work well inside Python Classes, these functions are independent from the Solver class.
They require the parameters of the solver (alpha, beta, etc. as arguments).
"""

@jit(cache=True)
def f(beta,theta, x):
    return np.tanh(beta * (x - theta))

@jit(cache=True)
def rhs(N_t,beta,theta, mat, v, x):
    # remve environment genes from the regulation matrix
    x_copy = x.copy()
    x_copy[:,:N_t] = 0

    # do the matrix multiplication to calculate the regulatory effect
    prod = x_copy @ mat.T
    prod = f(beta,theta,prod)

    # activation
    results = np.zeros_like(x)
    results[:,:prod.shape[1]]  = v[:,np.newaxis]* (prod - x[:,:prod.shape[1]])
    return results

@jit(cache=True)
def euler_ma_step(N_t,beta,theta,sigma, dt, x, v, mat, rng: np.random.Generator):
    dxdt_c = rhs(N_t,beta,theta,mat, v, x) * dt
    dxdt_s = np.zeros_like(dxdt_c)
    if not sigma == 0:

        dxdt_s[:,:mat.shape[0]] = rng.normal(0, sigma * np.sqrt(dt), (x.shape[0],mat.shape[0]))

    return dxdt_c + dxdt_s

@jit(cache=True)
def v_f(N_t,v_max,gamma,epsilon, x, t_values):
    x = x[:,:N_t]
    sum = np.sum(((x - t_values) / 2) ** 2,axis=1)
    return v_max * np.exp(-1 * gamma * sum)+epsilon

@jit(cache=True)
def integrate_internal_jit(N_t,v_max,gamma,epsilon,beta,theta,sigma,x_0, steps, dt, mat, target_values,rng):
    x_i = x_0

    # worker thread needs to have its own rng not based on the main thread:
    # If not (when using np.random.normal directly) this might causes a fringe case by chance where the np.random gets deadlocked
    # Since this np.random.normal will be executed 10^6 times in a single generation this causes a fail after 100 generations
    # only happens when using fork instead of spawn
    for i in range(steps):
        v = v_f(N_t,v_max,gamma,epsilon,x_i, target_values)
        x_i += euler_ma_step(N_t,beta,theta,sigma,dt, x_i, v, mat,rng)

    return x_i, v_f(N_t,v_max,gamma,epsilon,x_i, target_values)

@jit(cache=True)
def integrate_internal_history_jit(N_t,v_max,gamma,epsilon,beta,theta,sigma,x_0, steps, dt, mat, target_values,rng, x_hist = False):
    x_i = x_0  # dimensions: cells, gene
    v_s = np.empty(( steps,x_0.shape[0]))# dimensions: time,cells
    x_s = None
    if x_hist:
        x_s = np.empty((steps,x_0.shape[0],x_0.shape[1]))

    # worker thread needs to have its own rng not based on the main thread:
    # If not (when using np.random.normal directly) this might causes a fringe case by chance where the np.random gets deadlocked
    # Since this np.random.normal will be executed 10^6 times in a single generation this causes a fail after 100 generations
    # only happens when using fork instead of spawn

    for i in range(steps):
        v = v_f(N_t, v_max, gamma, epsilon, x_i, target_values)
        v_s[i,:] = v
        if x_hist:
            x_s[i,:,:] = x_i

        step = euler_ma_step(N_t, beta, theta, sigma, dt, x_i, v, mat, rng)
        x_i += step

    if x_hist:
        return x_s,v_s
    else:
        return x_i[np.newaxis,:,:], v_s