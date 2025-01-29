import itertools
import os
from itertools import permutations

import numpy as np
import matplotlib.pyplot as plt
import multiprocessing as mp

from Simulation import Solver
from tqdm.auto import tqdm
import pickle

from Old_Code.plot_cross_testing import output_path

defaults = {
    "beta": 20,
    "theta": 0.2,
    "v_max": 10,
    "gamma": np.log(10) / 2,
    "sigma": 0.25,
    "N": 40,
    "N_t": 8,
    "p1": 0.02,
    "p2": 0.02,
    "dt": 0.1,
    "generation_length": 8000,
    "early_stopping": False,
    "epsilon": 0.01,
}

T = 4000

matrix_path = "../example_mat.npy"
env_path = "../example_envs.pkl"
output_path = "order_test"

generation = 0
CONTROL = True

"""
tests effects of order of environments on the results
"""

if not os.path.exists(output_path):
    os.mkdir(output_path)

def work(args):
    x_0 = args[0]
    path = args[1]
    targets = args[2]
    id = args[3]

    mat = np.load(path)
    vs = []
    for env in targets:
        sol = Solver(defaults.copy())

        _, x, v = sol.integrate_jit(x_0, (0, T), 0.1, mat, env, "none")
        sol.sigma = 0
        x = x[:, :x_0.shape[1]]

        _, _, v = sol.integrate_jit(x, (0, 100), 0.01, mat, env, "growth-rate-only")

        x_0 = x[:,:x_0.shape[1]]

        v =np.mean(v[:,-1])
        vs.append(v)

    return id,vs

def plot(data,path):
    x_coordinates = [0,1,2,3]

    y_coordinates = list(data.keys())

    def sorting_key(key):
        return np.mean(data[key])
    y_coordinates.sort(key=sorting_key)

    # Initialize a 2D array (grid) to hold the scalar values
    grid = np.full((len(x_coordinates) , len(y_coordinates) ), np.nan)
    for y, perm_s in enumerate(y_coordinates):
        for x in x_coordinates:
            grid[x,y] = data[perm_s][x]



    #print(diagonal_dominance(grid))
    # Plotting the 2D heatmap
    fig,ax = plt.subplots(1,1,figsize=(5,25),tight_layout=True)

    img = ax.imshow(grid.T, origin='lower', cmap='plasma', interpolation='nearest')

    bar = fig.colorbar(label='Growth Rate',ax=ax,mappable=img)
    bar.set_label('Growth Rate',fontsize=18)
    bar.set_ticks(np.linspace(np.min(grid),np.max(grid),5), labels = [f"{i:.2}" for i in np.linspace(np.min(grid),np.max(grid),5)],fontsize=18)

    ticks =range(0, len(y_coordinates) )
    ax.set_yticks(ticks,labels=y_coordinates,fontsize=14,)

    space = np.linspace(np.min(grid),np.max(grid),100)
    colors = plt.cm.plasma(np.linspace(0,1,100))
    for t in ax.yaxis.get_ticklabels():
        key = t.get_text()
        v = np.mean(data[key])
        index = np.argmin(np.abs(space-v))
        t.set_color(colors[index])
    tick_strings = [f"$\\sigma_{i}$" for i in x_coordinates]
    ax.set_xticks(x_coordinates,tick_strings,fontsize=18)

    ax.set_ylabel('Permutation $\\sigma \\in S_4 $', fontsize=18)
    ax.set_title(f'Generation {generation}',fontsize = 20)
    plt.savefig(path+f"/{generation}_order.png")
    plt.show()

    fig,ax = plt.subplots(tight_layout=True)
    points = [np.mean(data[key]) for key in y_coordinates[::-1]]
    ax.scatter(range(len(points)),points)
    ax.tick_params(which = "major", axis= "both",labelsize="x-large")
    ax.set_ylabel("Growth Rate" ,fontsize="xx-large")
    ax.set_xlabel("Permutation Rank" ,fontsize="xx-large")
    #ax.set_xticks(range(len(points)),y_coordinates[::-1],fontsize=18,rotation=90)
    ax.set_ylim(0,10)
    plt.savefig(path + f"/{generation}_order2.png")
    plt.show()

if __name__ == "__main__":
    data = {}
    if os.path.exists(output_path+"/order_testing.pkl") and CONTROL:

        data = pickle.load(open(output_path + "/order_testing.pkl", "rb"))
        data_i = data.get(str(generation), None)
        if data_i is not None:
            plot(data_i, output_path)
            exit()

    if os.path.exists(env_path):
        targets = pickle.load(open(env_path, "rb"))
    else:
        raise FileNotFoundError





    x_0 = np.random.choice([-1.,1.],(200,40))
    #mat_path = MASTER_PATH+f"/strains/{generation}.npy"
    mat_path = matrix_path

    works = []
    for permut in permutations(list(range(len(targets)))):
        targets_p = [targets[i] for i in permut]
        ids = f"{permut[0]}{permut[1]}{permut[2]}{permut[3]}"
        works.append((x_0, mat_path, targets_p ,ids))

    pool = mp.Pool(8)
    pbar = tqdm(total=len(works))

    results = {}

    for id,vs in pool.imap(work,works):
        print(id,vs)
        results[id] = vs
        pbar.update(1)
    data[str(generation)] = results
    pickle.dump(data, open(output_path + "/order_testing.pkl", "wb"))
    pool.close()
    plot(data[str(generation)], output_path)