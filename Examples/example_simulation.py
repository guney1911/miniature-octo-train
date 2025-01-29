import multiprocessing as mp
import os
import pickle
import time
from multiprocessing import Pool

import numpy as np
from Simulation import Simulation, generate_prob_mat, get_vectors
from pprint import pprint

if __name__ == '__main__':
    mp.freeze_support()

    #settings for the simulation
    args = {"generation_length": 100,  "p1": 0.04, "p2": 0.04, "dt": 0.1, 'epsilon': 0.01,"sigma":0.22}

    #output location
    output_location = "test"
    #ensure that the location exists
    if not os.path.exists(output_location):
        os.mkdir(output_location)

    #location to save reg_mats
    matrix_save_subdirectory = output_location + "/ref-mats"

    #ensure that the location exists
    if not os.path.exists(matrix_save_subdirectory):
        os.mkdir(matrix_save_subdirectory)

    number_of_generations = 80 #number of generations
    number_of_target_environments = 4 #enviroments per generation
    number_of_strains = 20 #number of strains in the simulation
    number_of_copies = 30 #per strain

    if os.name == 'nt': # on windows
        pool = Pool(processes=10, maxtasksperchild=50)
    else: # on linux
        mp.set_start_method('forkserver') # is faster
        pool = Pool(processes=10, maxtasksperchild=40, initializer=os.nice, initargs=(20,))

    #mutation rates
    p_d = 0.006
    p_c = 0.001





    #load environments from pickle
    environments = pickle.load(open("environments.pkl", "rb"))
    environments = environments[:number_of_target_environments]

    # generate environments
    #get 4 vectors of len 8 and hamming distance between 2,4
    targets = get_vectors(number_of_target_environments, 8,2,4)
    #input vectors as defined
    inputs = np.array([[+1,-1,-1,-1,-1,-1],
                       [-1,+1,-1,-1,-1,-1],
                       [-1,-1,+1,-1,-1,-1],
                       [-1,-1,-1,+1,-1,-1],
                       ])

    #roll into single array
    environments = [(inputs[k],targets[k]) for k in range(number_of_target_environments)]


    for env in environments:
        print(f"LOADED ENV: {env[0]} {env[1]}")
    print("TARGETS:",environments)

    #save environments
    pickle.dump( environments,open(os.path.join(output_location, "environments.pkl"), "wb"))

    #create the simulation
    simulation = Simulation(args=args)

    #save simulation settings
    with open(os.path.join(output_location, "args.txt"), "w") as file:
        pprint(simulation.args, stream=file)

    #add strains to the simulation
    for i in range(number_of_strains):
        simulation.add_strain(N=number_of_copies)

    print(simulation)



    for i in range(number_of_generations):

        scores = None

        #shuffle the order of environments, is important for sigma = 0. In this case the network can become dependent on the order.
        np.random.shuffle(environments)

        #timekeeping
        t1 = time.time()

        ############## integrate the simulation ###################
        for k in range(number_of_target_environments):#for each environment

            #set the environment to the desired one
            simulation.target = environments[k]
            print("Current Target: ", k)

            t1_1 = time.time()

            #do the integration to the next generation
            simulation.generation(pool=pool)

            # do the (partial) fitness calculation
            scores = simulation.calculate_scores(pool, "multi_growth_rate_mean", period=number_of_target_environments)

            t2_2 = time.time()
            print(f"Done {p_d}:{p_c}: {i}-{k} in {t2_2 - t1_1}")

        #name of the best strain
        best_strain = max(scores, key=scores.get)
        print(f"Best Growth Rate: {scores.get(best_strain)}")

        #genertate the mutation probability matrix. If you want to adjust the mutation rate with the growth rate, do it here.
        prob_mat = generate_prob_mat(p_d,p_c)

        #this returns the regulatory matrices of the strains still in the simulation (all the strains that were not deleted + daughter strains) as an array (name, reg_mat)
        mat = simulation.evolution(pool, method ="best", scores=scores, child_strains=None, child_copies=number_of_copies, prob=prob_mat)

        #log the current growth rate
        with open(os.path.join(output_location, f"log.txt"), "a") as f:
            f.write(f"{i};{scores[best_strain]}\n")
        print(scores)

        #save the current best reg_mat
        for mati in mat:
            if mati[0] == best_strain:
                np.save(os.path.join(matrix_save_subdirectory, f"{i}.npy"), mati[1])
                break

        #you can also do the following to save all strains.
        #pickle.dump(mat, open(os.path.join(matrix_save_subdirectory, f"{i}.pkl"), "wb"))

        #or to save in a compressed format
        # np.savez_compressed(os.path.join(matrix_save_subdirectory, f"{i}.npz"), **{k[0]:k[1]for k in mat} )

        t2 = time.time()

        print(f"Done {p_d}:{p_c}: {i}: in {t2 - t1}")