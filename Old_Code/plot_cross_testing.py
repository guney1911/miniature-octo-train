import os

import numpy as np
import matplotlib.pyplot as plt
import multiprocessing as mp

from Simulation import Solver

from tqdm.auto import tqdm
import pickle
defaults = {
    "beta": 20,
    "theta": 0.2,
    "v_max": 10,
    "gamma": np.log(10) / 2,
    "sigma": 0,
    "N": 40,
    "N_t": 8,
    "p1": 0.02,
    "p2": 0.02,
    "dt": 0.1,
    "generation_length": 8000,
    "early_stopping": False,
    "epsilon": 0.01,
}

T = 500

matrix_path = "../example_mat.npy"
env_path = "../example_envs.pkl"
output_path = "cross_out"

CONTROL = False
generation = 0

"""
Test effects of mismatched inout genes and target genes.
untested code
"""


def work(args):
    x_0 = args[0]
    path = args[1]
    env = args[2]
    id = args[3]
    mat = np.load(path)
    sol = Solver(defaults.copy())




    t, x, v = sol.integrate_jit(x_0, (0, T), 0.1, mat, env, "none" )

    sol.sigma= 0
    x = x[:, :x_0.shape[1]]

    t, x, v = sol.integrate_jit(x, (0, 100), 0.01, mat, env, "none")
    v =np.mean(v[:,-1])

    return id,v

def plot(data,path):

    coordinates = [tuple(map(int, key.split('_'))) for key in data.keys()]
    max_x = max(coord[0] for coord in coordinates)
    max_y = max(coord[1] for coord in coordinates)

    # Initialize a 2D array (grid) to hold the scalar values
    grid = np.full((max_x + 1, max_y + 1), np.nan)

    # Fill the grid with scalar values
    for key, value in data.items():
        x, y = map(int, key.split('_'))
        grid[x, y] = value
    grid = grid
    control = grid[:,0]
    results = grid[:,1:]
    diagonal_sum = np.trace(results)
    print(results)
    print(diagonal_sum/4)
    #res1 = grid - control[:,np.newaxis]
    res2 = results - control[:,np.newaxis]

    #print(diagonal_dominance(grid))
    # Plotting the 2D heatmap
    fig,ax = plt.subplots(1,2,width_ratios=(1.2/6,4.8/6),sharey=True)
    left = ax[0].imshow(control[:,np.newaxis],origin='lower',cmap="magma",vmin = 0,vmax= 10,)
    bound = np.max(np.abs(res2))
    right = ax[1].imshow(res2, origin='lower', cmap='coolwarm', interpolation='nearest',vmax =bound,vmin=-1*bound  )
    fig.colorbar(label='$\\Delta v$',ax=ax,mappable=right ,location="right",fraction=0.031, pad=0.04)
    #fig.colorbar(label='Final Growth Rate',ax=ax,mappable=left ,location="left",fraction=0.033, pad=0.1)
    ticks =range(0, max_x+1)
    ax[1].set_xticks(ticks,labels=[f"Pat. {i+1}" for i in ticks])
    ax[0].set_xticks([0],labels=["Control"])
    ticks =range(0, max_y )
    ax[0].set_yticks(ticks,labels=[f"Pat. {i+1}" for i in ticks], rotation=90)
    #ax[0]
    #ax[1].yticks(ticks,labels=[f"Pattern {i}" for i in ticks])

    ax[0].set_ylabel('Target Pattern')
    ax[1].set_title("Difference Between Control and Input Pattern")
    #plt.ylabel('Input Pattern')
    fig.suptitle(f'Generation {generation}')
    plt.savefig(path+f"/{generation}_cross.png")
    plt.show()


if __name__ == "__main__":
    data = {}
    if os.path.exists(output_path + "/cross_testing.pkl") and CONTROL:

        data = pickle.load(open(output_path + "/cross_testing.pkl", "rb"))
        data_i = data.get(str(generation), None)
        if data_i is not None:
            plot(data_i, output_path)
            exit()

    if os.path.exists(output_path + "/targets.pkl"):
        targets = pickle.load(open(output_path + "/targets.pkl", "rb"))
    else:
        raise FileNotFoundError





    x_0 = np.random.choice([-1.,1.],(50,40))
    mat_path = matrix_path

    #mat_path = matrix_path+f"/strains/{generation}.npy" #if you have directory where you have saved different generation

    works = []
    for i in range(len(targets)): #environment
        for j in range(len(targets)+1): #input

            if j == 0:
                input = np.zeros_like(targets[0][0])
            else :
                input = targets[j-1][0]

            target = targets[i][1]
            work_arg = (x_0,mat_path,(input,target),f"{i}_{j}")
            works.append(work_arg)
    pool = mp.Pool(8)
    pbar = tqdm(total=len(works))

    results = {}

    for id,v in pool.imap(work,works):

        results[id] = v
        pbar.update(1)
    data[str(generation)] = results
    pickle.dump(data, open(output_path + "/cross_testing.pkl", "wb"))
    pool.close()
    plot(data[str(generation)],output_path)