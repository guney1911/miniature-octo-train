import os
import pickle

import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import savgol_filter
from tqdm import tqdm

from Simulation import Solver
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from tqdm.utils import envwrap

"""
plot various plots
untested code
"""
def get_hist(x_0,inputs,targets,mat,sigma = 0.25, t = 4000,dt=0.1):

    defaults = {
        "beta": 20,
        "theta": 0.2,
        "v_max": 10,
        "gamma": np.log(10) / 2,
        "sigma": sigma,
        "N": 40,
        "N_t": 8,
        "p1": 0.02,
        "p2": 0.02,
        "dt": 0.1,
        "generation_length": 8000,
        "early_stopping": False,
        "epsilon": 0.01,
    }
    x_hist = []
    v_hist = []
    v2_hist = []
    for env in zip(inputs,targets):
        sol = Solver(defaults.copy())

        _, x, v = sol.integrate_jit(x_0, (0, t), dt, mat, env, "both")

        x_0 = x[-1, :, :x_0.shape[1]]
        x = x[-1,:,:x_0.shape[1]]

        x_hist.append(x)
        v_hist.append(v)

        sol.sigma = 0
        _, _, v2 = sol.integrate_jit(x[-2], (0, 100), 0.01, mat, env, "both")
        v2_hist.append(v2[-1,:])

    return x_hist,v_hist,v2_hist

def unactivated_input(mat,x):
    defaults = {
        "beta": 20,
        "theta": 0.2,
        "v_max": 10,
        "gamma": np.log(10) / 2,
        "sigma": sigma,
        "N": 40,
        "N_t": 8,
        "epsilon": 0.01,
    }
    x_copy = x.copy()
    x_copy[:,:,  :defaults["N_t"]] = 0
    prod = x_copy @ mat.T
    return defaults["beta"] * (prod - defaults["theta"])


def jakobian(mat,x,target):
    defaults = {
        "beta": 20,
        "theta": 0.2,
        "v_max": 10,
        "gamma": np.log(10) / 2,
        "sigma": sigma,
        "N": 40,
        "N_t": 8,
        "epsilon": 0.01,
    }
    def v(x):
        x = x[:,:,:defaults["N_t"]]
        sum = np.sum(((x - target) / 2) ** 2,axis=2)
        # print(sum)
        return defaults["v_max"] * np.exp(-1 * defaults["gamma"] * sum)+defaults["epsilon"]

    def dvdx(x):
        d = np.zeros((x.shape[0],x.shape[1],46))
        d[:,:,:defaults["N_t"]] = -1* defaults["gamma"] * 0.5* (v(x) -defaults["epsilon"])[:, :, np.newaxis]*(x[:,:,:defaults["N_t"]] - target)
        return d

    def f(x):
        x_copy = x.copy()
        print(x_copy.shape)
        x_copy[:,:,:defaults["N_t"]] = 0
        prod = x_copy @ mat.T

        return np.tanh(defaults["beta"] * (prod - defaults["theta"]))


    jakobian_v = np.repeat(dvdx(x)[:, :, np.newaxis,: ],40,axis=2)


    jakobian_v = jakobian_v * (f(x) - x[:,:,:40])[:,:,:,np.newaxis]


    jacobian_f = defaults["beta"]*mat*(1-f(x)**2)[:,:,:,np.newaxis]

    return jakobian_v+ jacobian_f

MASTER_PATH = "output2_double_search_finetune_shuffle/0/0.005_0.005_0.006_0.006_5"

targets = pickle.load(open(MASTER_PATH + "/environments.pkl", "rb"))
N = 40
T = 500
dt = 0.01
sigma=0
generation = 79
control = False
growth_rate_hist = True
stability = False
dynamics = True
dynamics_full = False
dynamics_unactivated = False
pca = False
fourier = True

x_0_0 = np.random.choice([-1.,1.],(N,40))
mat = np.load(MASTER_PATH + f"/strains/{generation}.npy")

inputs = [t[0] for t in targets]
targets = [t[1] for t in targets]
if os.path.exists(MASTER_PATH + f"/{generation}_hist.pkl") and control:
    x_hist,v_hist,v2_hist,x2_hist = pickle.load(open(MASTER_PATH + f"/{generation}_hist.pkl", "rb"))
else:
    x_hist,v_hist,v2_hist = get_hist(x_0_0, inputs, targets, mat,sigma = sigma, t = T,dt=dt)
    x_hist = np.array(x_hist,dtype=np.float16)
    v_hist = np.array(v_hist,dtype=np.float16)
    v2_hist = np.array(v2_hist,dtype=np.float16)
    pickle.dump((x_hist,v_hist,v2_hist),open(MASTER_PATH + f"/{generation}_hist.pkl","wb"))

jakobians = []
if stability:
    for k in range(len(inputs)):
        x_downsampled = x_hist[k,::10000, :, :]
        j = jakobian(mat,x_downsampled,targets[0])
        jakobians.append(j)

if growth_rate_hist:
    fig,ax = plt.subplots()
    colors = plt.cm.Set1(np.linspace(0,1,N))
    for i in range(N):
        v_filtered = savgol_filter(v_hist[:,:,i].flatten(), int(10 / dt), 1)
        ax.plot(np.arange(0, len(targets)*T,dt), v_filtered,color=colors[i])
    for k in range(len(targets)):
        color = "gray" if k %2 == 0 else "white"
        ax.axvspan(k * T, (k + 1) * T, alpha=0.3, color=color)
    res = plt.waitforbuttonpress()
    plt.close()
    if res:
        exit()

if pca:
    fig,axs = plt.subplots(2,2, figsize=(20,20))#,subplot_kw=dict(projection='3d'))
    axs = axs.flatten()
    for k in range(len(targets)):
        print(x_hist.shape)
        x_filtered =   savgol_filter(x_hist[k,:,:,:8], int(10 / dt), 1,axis=0)
        x_downsampled = x_filtered[::1000,:,:]
        original = x_downsampled.shape
        print(original)
        x_downsampled = x_downsampled.reshape(-1,8)
        print(x_downsampled.shape)

        pca= PCA(n_components=2)

        x_i_i =  StandardScaler().fit_transform(x_downsampled)
        principalComponents = pca.fit_transform(x_i_i)
        principalComponents = principalComponents.reshape(original[0],original[1],2)
        print(principalComponents.shape)
        ax = axs[k]
        colors = plt.cm.magma(np.linspace(0,1,original[0]))
        for j in range(original[1]):
            ax.scatter(principalComponents[:,j,0],principalComponents[:,j,1],color=colors)
    res = plt.waitforbuttonpress()
    plt.close()
    if res:
        exit()

for i in range(N):
    print("Final Growth Rate ", v2_hist[:,i])
    if dynamics:
        fig,axs = plt.subplots(2,4, figsize=(20,10),sharex=True,sharey=True)
        axs = axs.flatten()
        for j, ax in enumerate(axs):
            t = np.arange(0, len(targets)*T,dt)
            ax.plot(t,x_hist[:,:,i,j].flatten(),alpha=0.5,color="black")
            x_filtered = savgol_filter(x_hist[:,:,i,j].flatten(),int(10/dt),1)
            ax.plot(t,x_filtered,color="black")
            ax.set_ylim(-2,2)
            ax.set_title(f"Gene {j}",fontsize="x-large")
            ax.set_ylabel("Expression")
            ax.set_xlabel("Time")
            for k in range(len(targets)):
                color = "green" if targets[k][j] >0 else "red"
                ax.axvspan(k*T,(k+1)*T,alpha=0.3,color=color)
        fig.suptitle(f"Generation {generation}",fontsize="xx-large")
        res = plt.waitforbuttonpress()
        plt.close()
        if res:
            exit()

    if dynamics_full:
        fig,axs = plt.subplots(8,5, figsize=(15,24),sharex=True,sharey=True)
        axs = axs.flatten()
        for j, ax in enumerate(axs):
            t = np.arange(0, len(targets)*T,dt)
            ax.plot(t,x_hist[:,:,i,j].flatten(),alpha=0.5,color="black")
            x_filtered = savgol_filter(x_hist[:,:,i,j].flatten(),int(10/dt),1)
            ax.plot(t,x_filtered,color="black")
            ax.set_ylim(-2,2)
            ax.set_title(f"Gene {j}",fontsize="x-large")
            ax.set_ylabel("Expression")
            ax.set_xlabel("Time")
            for k in range(len(targets)):
                if j < 8:
                    color = "green" if targets[k][j] >0 else "red"
                else:
                    color = "gray" if k % 2 == 0 else "white"
                ax.axvspan(k*T,(k+1)*T,alpha=0.3,color=color)
        fig.suptitle(f"Generation {generation}",fontsize="xx-large")
        res = plt.waitforbuttonpress()
        plt.close()
        if res:
            exit()
    if dynamics_unactivated:
        fig, axs = plt.subplots(8, 5, figsize=(15, 24), sharex=True, sharey=True)
        axs = axs.flatten()
        x_unactivated = unactivated_input(mat,x_hist[:,:,i,:])
        print(x_unactivated.shape)
        for j, ax in enumerate(axs):
            t = np.arange(0, len(targets) * T, dt)
            ax.plot(t, x_unactivated[:, :, j].flatten(), alpha=0.5, color="black")
            x_filtered = savgol_filter(x_unactivated[:, :,  j].flatten(), int(10 / dt), 1)
            ax.plot(t, x_filtered, color="black")
            ax.set_ylim(-50, 50)
            ax.set_title(f"Gene {j}", fontsize="x-large")
            ax.set_ylabel("Expression")
            ax.set_xlabel("Time")
            for k in range(len(targets)):
                if j < 8:
                    color = "green" if targets[k][j] > 0 else "red"
                else:
                    color = "gray" if k % 2 == 0 else "white"
                ax.axvspan(k * T, (k + 1) * T, alpha=0.3, color=color)
        fig.suptitle(f"Generation {generation}", fontsize="xx-large")
        res = plt.waitforbuttonpress()
        plt.close()
        if res:
            exit()
    if stability:
        fig,axs = plt.subplots(2,2)
        axs = axs.flatten()
        for k in range(len(targets)):
            ax = axs[k]
            matricies = jakobians[k][:,i,:,:]
            print(matricies.shape[0])
            eigv = np.linalg.eigvals(matricies[:,:,:40])

            colors = plt.cm.magma(np.linspace(0,1,matricies.shape[0]))
            for j in [2]:
                ax.scatter(range(len(eigv[j])), np.sort(np.real(eigv[j])), color=colors[j])
        res = plt.waitforbuttonpress()
        plt.close()
        if res:
            exit()

    if fourier:
        fig,axs = plt.subplots(2,4, figsize=(20,10))
        axs = axs.flatten()
        for j, ax in enumerate(axs):
            for k in range(len(targets)):
                freq_val = np.fft.fft(x_hist[k,:,i,j])
                freq = np.fft.fftfreq(len(x_hist[k,:,i,j]))
                ax.plot(freq,np.abs(freq_val))
                ax.set_ylim(0,1000)
        res = plt.waitforbuttonpress()
        plt.close()
        if res:
            exit()
