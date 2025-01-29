import multiprocessing as mp
import os
import pickle
import time
from multiprocessing import Pool

import numpy as np
from Simulation import Simulation,generate_prob_mat
from pprint import pprint

"""
run a grid search with parameters: number of environments, mutation rates, sigma values 
"""
if __name__ == '__main__':
    mp.freeze_support() # must be the first line!

    simulation_args = {"generation_length": 4000, "p1": 0.04, "p2": 0.04, "dt": 0.1, 'epsilon': 0.01}

    output_path = "output_triple_search"

    matrix_save_subdirectory = "strains"

    environment_location = "../example_envs.pkl"

    trials_per_grid = 3#
    number_of_generations = 80#
    strains = 20
    copies = 30

    if os.name == 'nt': #on windows
        pool = Pool(processes=10, maxtasksperchild=50)
    else: # on cluster
        mp.set_start_method('forkserver')
        pool = Pool(processes=10, maxtasksperchild=40, initializer=os.nice, initargs=(20,))

    ###########Grid Search Locations############
    mutation_rates = [(0.005, 0.006),
                      (0.003,0.004),
                      (0.006,0.001)] #(p_d,p_c)

    sigmas = [0,0.05,0.1,0.2,0.25]#
    number_of_environments = [1, 2, 3, 4]

    if not os.path.exists(output_path):
        os.mkdir(output_path)

    for number_of_env in number_of_environments:
        for sigma in sigmas:
            #make a copy
            args = simulation_args.copy()
            args["sigma"] = sigma

            if sigma == 0:
                args["generation_length"] = 500

            master_output = os.path.join(output_path, str(number_of_env), str(sigma)) #output location dependent on grid params

            if not os.path.exists(os.path.join(output_path, str(number_of_env))):
                os.mkdir(os.path.join(output_path, str(number_of_env)))

            if not os.path.exists(master_output):
                os.mkdir(master_output)

            with open(os.path.join(master_output, "mutation_rates.pkl"), "wb") as f:
                pickle.dump(mutation_rates, f)

            existing = os.listdir(master_output) #get already done mutation rates
            print(existing)

            for pd,pc in mutation_rates:

                print(pd,pc)

                for j in range(trials_per_grid):

                    output = os.path.join(master_output, f"{pd}_{pc}_{j}")

                    if f"{pd}_{pc}_{j}" in existing:
                        print("Skipping:", f"{pd}_{pc}_{j}")
                        continue #skip this mutation rate

                    if not os.path.exists(output):
                        os.mkdir(output)

                    matrix_save_location = os.path.join(output, matrix_save_subdirectory)
                    if not os.path.exists(matrix_save_location):
                        os.mkdir(matrix_save_location)

                    environments = pickle.load(open(environment_location, "rb")) #load environment
                    environments = environments[:number_of_env]


                    simulation = Simulation(args=args)

                    with open(os.path.join(output, "args.txt"), "w") as file:
                        pprint(simulation.args, stream=file)



                    print("environments:", environments)
                    pickle.dump(environments, open(os.path.join(output, "environments.pkl"), "wb"))

                    for i in range(strains):
                        _ = simulation.add_strain(N=copies)
                    print(simulation)




                    for i in range(number_of_generations):

                        scores = None
                        t1 = time.time()

                        np.random.shuffle(environments)

                        for k in range(number_of_env):
                            simulation.environment = environments[k]
                            t1_1 = time.time()
                            print("Target: ", k)
                            simulation.generation(pool=pool)
                            scores = simulation.calculate_scores(pool, "multi_growth_rate_mean",period=number_of_env)
                            t2_2 = time.time()

                            print(f"Done {pd}_{pc}: {j} : {i}: {k} in {t2_2 - t1_1}")

                        best_strain = max(scores, key=scores.get)
                        current_growth_rate = scores.get(best_strain)


                        print(f"CURRENT GROWTH RATE: {current_growth_rate}")
                        prob = generate_prob_mat(pd,pc)

                        mat = simulation.evolution(pool, "best", scores=scores, child_strains=None, child_copies=copies, prob=prob)

                        with open(os.path.join(output, f"log.txt"), "a") as f:
                            f.write(f"{i};{scores[best_strain]}\n")


                        for mati in mat:
                            if mati[0] == best_strain:
                                np.save(os.path.join(matrix_save_location, f"{i}.npy"), mati[1])
                                break

                        t2 = time.time()
                        print(f"Done {pd}_{pc}: {j} : {i}: in {t2 - t1}")

