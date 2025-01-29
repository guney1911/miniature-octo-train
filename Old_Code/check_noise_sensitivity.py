import pickle
import time

from Simulation import Solver
import numpy as np
import os
import multiprocessing as mp

"""
check senstitivity to noise
untested
"""

def work(kwargs):
    path = kwargs['path']
    generation = kwargs['generation']
    mat = np.load(path)

    environment = kwargs['environment']
    x_0 = kwargs['x_0']
    sigma = kwargs['sigma']


    solv = Solver()
    solv.sigma = sigma


    t, x, v = solv.integrate_jit(x_0, (0, 4000), 0.1, mat, environment)
    x = x[:, :x_0.shape[1]]
    solv.sigma = 0
    _, x2, v2 = solv.integrate_jit(x, (0, 100), 0.01, mat, environment)
    return generation,sigma,np.mean(v2),np.std(v2)

if __name__ == "__main__":
    if os.name == 'nt':
        pool = mp.Pool(processes=7, maxtasksperchild=100)
    else:
        mp.set_start_method('forkserver')
        pool = mp.Pool(processes=7, maxtasksperchild=100, initializer=os.nice, initargs=(20,))

    environments = pickle.load(open("../example_envs.pkl","rb"))
    x_0 = np.random.choice([-1,1], (50,40))
    output_path = "noise_results"

    period = 4
    sigmas = np.linspace(0,0.5,20)
    generations = [0,2,4,5,8,10,12,15,20,25,30,35,40,45,50,55,60,65,70,75,79]

    paths = ["output2_triple_search/4/0/0.003_0.003_0.004_0.004_0/strains/{}.npy", #path to evolution results {} gets replaced by the generation
             "output2_triple_search/4/0.05/0.006_0.006_0.001_0.001_1/strains/{}.npy",
             "output2_triple_search/4/0.1/0.006_0.006_0.001_0.001_1/strains/{}.npy",
             "output2_triple_search/4/0.25/0.006_0.006_0.001_0.001_1/strains/{}.npy",]

    outputs = ["noise_dependence_4_0_0.003_0.004_1_{}.txt", #save name {} gets replaced by the generation
               "noise_dependence_4_0.05_0.006_0.001_1_{}.txt",
               "noise_dependence_4_0.1_0.006_0.001_1_{}.txt",
               "noise_dependence_4_0.25_0.006_0.001_1_{}.txt",
               ]



    environments = environments[:period]

    if not os.path.exists(output_path):
        os.mkdir(output_path)
    outputs = [os.path.join(output_path, output) for output in outputs]

    for moutput,path in zip(outputs,paths):
        for N in [1,2,3]:#range(len(environments)):
            t1 = time.time()
            environment = environments[N]
            output = moutput.format(N)
            print(output)
            print()
            works = []

            for sigma in sigmas:
                for generation in generations:
                    settings = {
                         "path": path.format(generation),
                         "generation": generation,
                         "environment":environment ,
                         "sigma": sigma,
                         "x_0": x_0,
                    }
                    works.append(settings)

            for generation,sigma,v2,vs in pool.imap(work, works,chunksize=int(len(works)/7)+1):
                with open(output, "a") as f:
                    f.write(f"{generation};{sigma};{v2};{vs}\n")

            t2 = time.time()
            print(t2-t1)