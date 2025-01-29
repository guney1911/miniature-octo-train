
from multiprocessing import Pool

import numpy as np

from Simulation.sim_helpers import generate_random_string,mutation

from Simulation.strain import Strain
from Simulation.solver import Solver


def _generation_mapping(arguments):
    """
    internal used for multithreading
    """
    args = arguments[0]
    _input = arguments[1][0]
    target = arguments[1][1]
    strain = arguments[2]

    steps = int(args["generation_length"] / args["dt"])
    solver = Solver(args=args)
    x_0 = np.zeros((strain.state.shape[0],strain.state.shape[1]+_input.shape[0]))
    x_0 [:,:strain.state.shape[1]] = strain.state
    x_0[:,strain.state.shape[1]:] = _input
    x_i, v_s = solver._integrate_internal_hist_jit(x_0, steps, args["dt"], strain.reg_mat, target, x_hist=False)

    strain.update_v_history(v_s)
    strain.state= x_i[:,:strain.state.shape[1]]

    return strain

def _generation_mapping_fast(arguments):
    """
    internal used for multithreading
    """
    args = arguments[0]
    _input = arguments[1][0]
    target = arguments[1][1]
    strain = arguments[2]

    steps = int(args["generation_length"] / args["dt"])
    solver = Solver(args=args)
    x_0 = np.zeros((strain.state.shape[0],strain.state.shape[1]+_input.shape[0]))
    x_0 [:,:strain.state.shape[1]] = strain.state
    x_0[:,strain.state.shape[1]:] = _input

    x_i,v_i = solver._integrate_internal_jit(x_0, steps, args["dt"], strain.reg_mat, target)

    strain.state= x_i[:,:strain.state.shape[1]]
    strain.avg_v = np.mean(v_i)
    return strain




class Simulation:
    def __init__(self, args=None, environment=None, ):
        """
        Class for the simulator
        :param args: arguments for the simulation. Overrides defaults
        :param environment: environment for the simulation. Should be tuple ( input_vec, target_state)
        """
        # Default Settings
        defaults =  {
            "beta": 20,
            "theta": 0.2,
            "v_max": 10,
            "gamma": np.log(10) / 2,
            "sigma": 0.25,
            "N": 40,
            "N_t": 8,
            "N_i": 6,
            "p1":0.02,
            "p2":0.02,
            "dt":0.01,
            "generation_length":4000,
            "epsilon": 0.01,
        }
        # Merge default settings with arguments
        if args is not None:
            args = {**defaults, **args}
        else:
            args = defaults
        #choose a random environment if none are given
        if environment is None:
            inputs= np.zeros(args["N_i"])
            inputs[0] = 1
            environment =  (inputs,np.random.choice([-1,1],args["N_t"]))

        if len(environment[0]) != args["N_i"]:
            raise ValueError(f"Input vector should have length N_i ({len(environment[0])} vs {args["N_i"]})")

        if len(environment[1]) != args["N_t"]:
            raise ValueError(f"Input vector should have length N_i ({len(environment[1])} vs {args["N_i"]})")


        #set settings
        self.environment = (np.array(environment[0]),np.array(environment[1]))
        self.args = args
        self.strains = []
        self.strain_names= []
        self.score_store = {}
        self.generation_no = 0
        self.last_count = -1
        self.last_Count_state = None

    def add_strain(self,N=10,reg_mat = None):
        """
        Add a strain to the simulation.
        :param N: Number of copies of this strain
        :param reg_mat: Optional: Regulatory matrix (if non provided a random matrix will be created)
        :return:  Nothing, Updates the simulation state.
        """
        #create an empty timeline to help with plotting
        #empty =  int(self.generation_no * self.args["generation_length"]/self.args["dt"])
        # generate the strain name randomly
        strain_name = generate_random_string(self.strain_names)
        # create an empty cell
        strain = Strain(N,N_t = self.args["N_t"], N_i = self.args["N_i"],N= self.args["N"], p1=self.args["p1"],p2 =  self.args["p2"],name=strain_name,reg_mat=reg_mat)
        # replace the history of the cell with the empty one
        #strain.update_v_history(np.zeros((N,empty)))
        #save strain and cell
        self.strain_names.append(strain_name)
        self.strains.append(strain)


    def generation(self,pool: Pool):
        """
        Iterate the simulation to the next generation.
        :param pool: Multithreading pool
        :return: Nothing, Updates the simulation state.
        """

        chunk_size = round(len(self.strains)/pool._processes)
        chunk_size = chunk_size if chunk_size > 1 else 1
        print("Chunksize: ", chunk_size)

        # calculate the state without history
        iter = pool.imap_unordered(_generation_mapping_fast, [(self.args, self.environment, strain) for strain in self.strains if strain.alive], chunksize=chunk_size)
        result = []
        for i in iter:
            result.append(i)
        self.strains =result + [strain for strain in self.strains if not strain.alive]
        self.generation_no = self.generation_no + 1

    def calc_final_v(self,steps, pool: Pool):
        """
        Calculates the growth rate of strains with their current states in the simulation without noise.
        :param steps: Number of steps for the noiseless calculation. Step size is 0.01s.
        :param pool: multithreading pool
        :return: returns a list of strain objects.
            The history of the strain objects contains the noiseless growth rate history.
        """
        args = self.args.copy()
        args['generation_length'] = steps
        args['sigma'] = 0
        args["early_stopping"] = False
        args["dt"] = 0.01
        result = pool.map(_generation_mapping, [(self.args, self.environment, strain) for strain in self.strains])
        return result


    def calculate_scores(self, pool: Pool,method,**args):
        """
        Calculates fitness scores for the strains and their current states in the simulation.
        Can calculate scores over multiple generations.

        :param pool: Multiprocessing pool
        :param method: "single_growth_rate"
                        Uses the latest growth rates to calculate the scores
                        "multi_growth_rate_mean"
                        Uses the average of the latest growth rates. requires period keyword
        :keyword period: Number of generations to average over
        :keyword timeframe: timeframe for the final noiseless growth rate calculation. Defaults to 100s
        :return: Either scores or None. None indicates that no evolution should take place. Scores is adict of strain names and scores.
        """
        scores = {}
        timeframe = args.get("timeframe",100)

        if method == "single_growth_rate": # method to use only the latest generation

            array = self.calc_final_v(timeframe, pool) #get the noiseless growth rates for each strain
            for i in array:
                scores[i.name] = np.mean(i.v_history[:, -1]) # save the mean of the growth rate across copies to the dict.
            return scores

        elif method == "multi_growth_rate_mean": #method to use multiple generations.

            period = args["period"] # number of generations to use

            array = self.calc_final_v(timeframe, pool)

            if self.score_store.get("number_of_scores",0) == period-1: # if we have seen the required number of generations return
                for i in array:
                    # get the average of copies and add the result to the average over generations
                    scores[i.name] = self.score_store.get(i.name,0) + np.mean(i.v_history[:, -1])/period

                self.score_store = {"number_of_scores": 0}  # reset store
                return scores

            else:
                for i in array:
                    self.score_store[i.name] = self.score_store.get(i.name,0) + np.mean(i.v_history[:, -1])/period # add to the store
                self.score_store["number_of_scores"] = self.score_store.get("number_of_scores",0) + 1 # increment
                return None

        elif method == "multi_growth_rate_min":
            period = args["period"]
            array = self.calc_final_v(timeframe, pool)

            if self.score_store.get("number_of_scores",0) == period-1: # if we have reached the period return the avg. of the previous scores
                for i in array:
                    scores[i.name] = min(self.score_store.get(i.name,float('inf')), np.mean(i.v_history[:, -1]))

                self.score_store = {"number_of_scores": 0}  # reset store
                return scores

            else:
                for i in array:
                    self.score_store[i.name] = min(self.score_store.get(i.name,float('inf')) , np.mean(i.v_history[:, -1]))# add to the store
                self.score_store["number_of_scores"] = self.score_store.get("number_of_scores",0) + 1 # increment
                return None

        elif method == "template_multi": # template for multi-generation calculation.
            #TODO: remove this line after implementing the method!
            raise NotImplementedError("You have to implement this!")

            period = args["period"]  # number of generations to use

            array = self.calc_final_v(timeframe, pool)

            # if we have seen the required number of generations return scores
            if self.score_store.get("number_of_scores",0) == period - 1:
                for i in array:
                    strain_name = i.name
                    previous_data = self.score_store.get(strain_name, 0)
                    current_history = i.v_history
                    #TODO: implement your own method to calculate scores from previous calculations and current history
                    scores[i.name] = None

                self.score_store = {"number_of_scores": 0}  # reset store
                return scores

            else:
                for i in array:
                    strain_name = i.name
                    previous_data = self.score_store.get(strain_name, 0)
                    current_history = i.v_history
                    #TODO: implement your own calculation to be used by the next calculations from previous calculations and current history.
                    self.score_store[strain_name] = None

                self.score_store["number_of_scores"] = self.score_store.get("number_of_scores", 0) + 1  # increment
                return None

        if method == "template_single":  # method to use only the latest generation
            #TODO: remove this line after implementing the method!
            raise NotImplementedError("You have to implement this!")

            array = self.calc_final_v(timeframe, pool)
            for i in array:
                strain_name = i.name
                current_history = i.v_history
                #TODO: Implement your own calculation based ob the current_history
                scores[strain_name] = None
            return scores
        else:
            raise NotImplementedError(f"{method} not implemented!")



    def evolution(self,pool: Pool,method,scores = None,return_reg_mat= True,child_strains = None,child_copies=20,**kwargs):

        """
        This functions resets the simulation history!
        1) removes cells that are under the selection criteria.
        2) from the cell with the highest criteria, creates children that mutated.
        3) starts from zero

        :param pool: Multithreading pool
        :param method: Selection Criteria. Currently Available:

        "threshold": all cells with growth rate less than threshold are removed.
                    The cell with the maximum growth rate generates children.

        "best": best performing cell is chosen

        :param child_strains: Number of child strains to create for each deleted strain. Set to None to create as many
                                children as the number of dead strains per fit strain.
        :param child_copies: Number of copies per created child.

        :param kwargs: following keyowods
        :keyword threshold: Controls the threshold for the threshold method
        :keyword probs: Controls mutation probabilities. Should be a prob. matrix as in sim_helper.generate_prob_mat

        :return:
        """


        if scores is None:
            raise ValueError


        to_delete = []
        children = []

        if method == "threshold":
            best_strain= max(scores, key=scores.get)
            children.append(best_strain)
            for key in scores.keys():
                if key == best_strain:
                    continue
                if scores[key] < kwargs["threshold"]:
                    to_delete.append(key)

            if len(to_delete) == 0:
                to_delete.append(min(scores, key=scores.get))

        elif method == "best":
            best_strain = max(scores, key=scores.get)
            children.append(best_strain)
            to_delete = [key for key in scores.keys() if not key == best_strain]

        elif method == "template":
            #TODO: Remove this line after implementation
            raise NotImplementedError("You have to implement this!")

            for name,fitness_score in scores:
                #TODO: Implement your own method
                children.append(name) # if this strain will have offspring
                to_delete.append(name) # if this strain will be deleted will go extinct.

                # if a strain is both in children and to_delete, it will have offspring and go extinct.
                # if it is in neither, it will simply keep on existing.

        else:
            raise NotImplementedError(f"{method} not implemented!")

        print("Strains that will be deleted",to_delete)
        print("Strains that will create daughter strains",children)

        parent_reg_matrices=[] #list of parent reg matricies

        new_strains = [] #strains that will be still in the simulation (parent strains)

        for strain in self.strains:
            #find reg matrices
            if strain.name in children:
                parent_reg_matrices.append((strain.name,strain.reg_mat))
                children.remove(strain.name)

            #reset the cells that should continue living, disregard cells that should be deleted
            if not strain.name in to_delete:
                strain.reset()
                new_strains.append(strain) #

        #print("child mat",parent_reg_matrices)
        # if the number of daughters is not specified creates as many as deleted strains.
        if child_strains is None:
            child_strains = int(len(to_delete)/len(parent_reg_matrices))

        #generate new daughters for parent strains that should create offspring.

        for name,reg_mat in parent_reg_matrices: # for each parent strain
            for i in range(child_strains): #create kwargs["child_strains"] offspring
                source_strain = name

                exisiting_daughters = [s for s in self.strain_names if s.startswith(source_strain)]
                exisiting_daughter_ids = [s.split('_')[-1] for s in exisiting_daughters if len(s.split('_'))>1]
                daughter_id = generate_random_string(exisiting_daughter_ids,2)


                old_mat = reg_mat
                new_mat = mutation(old_mat,kwargs["prob"])


                print(f"strain {name}_{daughter_id}: changes: {np.count_nonzero(old_mat-new_mat)}")

                #crate a new strain object
                new_strain = Strain(child_copies,N=self.args["N"],name=f"{name}_{daughter_id}",reg_mat=new_mat)
                new_strains.append(new_strain)
                self.strain_names.append(f"{name}_{daughter_id}")

        #remove old strains from the list of names
        for strain in to_delete:
            self.strain_names.remove(strain)

        self.strains = new_strains

        if return_reg_mat:
            return parent_reg_matrices



    def __str__(self):
        S = "Simulation:\n"
        for i in self.strains:
            S += f"{i.name}: {i.cells} : average growth rate: {i.avg_v:.2f} alive: {i.alive}\n"
        return S



