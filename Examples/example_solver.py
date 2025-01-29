import time
import pickle
from matplotlib import pyplot as plt
from Simulation import Solver,generate_reg_matrix
import numpy as np
from tqdm.auto import tqdm

copies = 20
size = 40
N_t = 8
N_i = 6
T = (0,4000)
dt =0.1
x_0 = np.random.choice([-1,1],(copies,size)) #starting point

# set environment and load matrix
_input,target = pickle.load(open("../example_envs.pkl", "rb"))[0]
#_input = np.random.choice([-1,1],N_i)
#target = np.random.choice([-1,1],N_t)

mat = np.load("../example_mat.npy")
#mat = generate_reg_matrix(size,N_t,N_i,[0.3,0.3])

#you can add solver arguments here
solver_args = {
    #"sigma":0
}
#create a solver
slv = Solver(args=solver_args)

#integrate without history
t1 = time.perf_counter()
t,x,v = slv.integrate(x_0,T,dt,mat,(_input,target),hist="none")
t2 = time.perf_counter()

print(t2-t1)




#integrate with only growth rate history
t1 = time.perf_counter()
t,x,v = slv.integrate(x_0,T,dt,mat,(_input,target),hist="growth-rate-only")
t2 = time.perf_counter()

print(t2-t1)
plt.figure()
for i in range(copies):
    plt.plot(t,v[:,i])

plt.show()


#integrate with full history
t1 = time.perf_counter()
t,x,v = slv.integrate(x_0,T,dt,mat,(_input,target),hist="both")
t2 = time.perf_counter()

print(t2-t1)
plt.figure()
for i in range(copies):
    plt.plot(t,v[:,i])

plt.show()

###################### Examples for Progress Bar ##############################

# a single progress bar
pbar = tqdm(total=int( (T[1] - T[0])/dt ))
t1 = time.perf_counter()
t,x,v = slv.integrate(x_0,T,dt,mat,(_input,target),hist="growth-rate-only",pbar = pbar)
t2 = time.perf_counter()

print(t2-t1)
plt.figure()
for i in range(copies):
    plt.plot(t,v[:,i])

plt.show()

#progress bar that is persistent over multiple integrations
pbar = tqdm(total=int( (T[1] - T[0])/dt ) *2)
t1 = time.perf_counter()
t,x,v = slv.integrate(x_0,T,dt,mat,(_input,target),hist="growth-rate-only",pbar = pbar)
t2 = time.perf_counter()

print(t2-t1)
plt.figure()
for i in range(copies):
    plt.plot(t,v[:,i])

plt.show()

t1 = time.perf_counter()
t,x,v = slv.integrate(x_0,T,dt,mat,(_input,target),hist="growth-rate-only",pbar = pbar)
t2 = time.perf_counter()

print(t2-t1)
plt.figure()
for i in range(copies):
    plt.plot(t,v[:,i])

plt.show()

###################### JIT Versions ###########################################


t1 = time.perf_counter()
t, x, v = slv.integrate_jit(x_0, T, dt, mat, (_input, target), hist="none")
t2 = time.perf_counter()

print(t2 - t1)

t1 = time.perf_counter()
t, x, v = slv.integrate_jit(x_0, T, dt, mat, (_input, target), hist="growth-rate-only")
t2 = time.perf_counter()

print(t2 - t1)
plt.figure()
for i in range(copies):
    plt.plot(t, v[:, i])

plt.show()

t1 = time.perf_counter()
t, x, v = slv.integrate_jit(x_0, T, dt, mat, (_input, target), hist="both")
t2 = time.perf_counter()

print(t2 - t1)
plt.figure()
for i in range(copies):
    plt.plot(t, v[:, i])

plt.show()
