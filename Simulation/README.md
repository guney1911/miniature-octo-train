# Simulation Package
Güney Erin Tekin
contact: gtekin@student.ethz.ch guney.erin@gmail.com

This package contains a simulation based on a noise-driven adaptation model. 
- The [solver.py](solver.py) (and [jit version](jit_solver_helpers.py)) implement the model and a solver based on Euler-Maruyama to solve it. 
- The [simulation.py](simulation.py) implements a simulation based the solver to simulate evolution based on the regulatory matrices in the model.
- The files [strain.py](strain.py) and [sim_helpers.py](sim_helpers.py) contain helper functions. 

Apart from this documentation file, all of the functions are very well commented. When in doubt please refer to the source code.


## Solver
The solver is based on the Euler-Maruyama method. The basic usage is:

~~~python
from Simulation import Solver,generate_reg_matrix

import numpy as np


#vreate a solver object
slv = Solver(args = {"sigma":0.25,})

#create a random regulatory matrix with 0.2 probability for -1 and 0.2 probability for 1
mat = generate_reg_matrix(40,8,6,[0.2,0.2])

#generate 30 random starting points
x_0 = np.random.choice([-1,1],(30,40))

#set the target state for maximal growth rate
target = np.ones(8)

#set the input gene
input = np.ones(6)

#integrate 
t,x,v = slv.integrate_jit(x_0,(0,4000),0.01,mat,(input,target),hist="both")
~~~

### Instantiating the class
For instantiating the solver there is only a single argument `args` which is a dictionary. 
It overrides the default values of the model parameters these values are shown below. 

| argument | default value | definition                                                                                                                                                                          |
|----------|---------------|-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| beta     | 20            | model Constant                                                                                                                                                                      |
| theta    | 0.2           | model constant                                                                                                                                                                      |
| v_max    | 10            | model constant                                                                                                                                                                      |
| gamma    | ln(10)/2      | model constant                                                                                                                                                                      |
| sigma    | 0.25          | model constant                                                                                                                                                                      |
| N        | 40            | number of genes                                                                                                                                                                     |
| N_t      | 8             | number of "target" genes. The first N_t genes are designed as target. (i.e if we use the default values, there will be 40 genes and the first 8 of them will be the "target" genes. |
| epsilon  | 0.01          | model constant                                                                                                                                                                      |

### Integrating the model
The model can be integrated by the `integrate` or `integrate_jit` functions. These are functionally equivalent but the __jit_ version is compiled during the runtime. 
This causes a 10x speedup. You can control the level of saved history with the `hist` option. If set to `hist = "none"`, no history is kept. 
In this case, the function returns only the latest state and the growth rate. 
This is used for the evolution simulation. 
Other settings (`hist = "growth-rate-only"` or `hist = "both"`) allow for integration with growth rate history or integration with all histories, but these are much more time and memory consuming.

The solver can integrate from multiple starting points at the same time. This is implemented in a vectorized fashion and benefits immensely from NumPy (BLAS) optimizations. 
This is a very good way to speed up calculations when integrating for multiple starting points. 
We suggest integrating at least 20-30 starting points at the same time. 
This only causes a minor performance hit (1-2x) compared to integrating a single starting point.
If the number of starting points is too much, the NumPy will spawn new threads to automatically parallize the integration. 
This might be undesirable in certain cases. 


### Just-In-Time Compilation

We use `numba` to accelerate our integration even further. The numba library can automatically compile python code to c++ and then to machine code.
A big bottleneck for the calculation is the main `for` loop for the integration, so we use numba to compile this for loop. 
This can also increase the speed of the integration (2-3x) depending on the number of steps.
The compilation itself takes longer than the calculation, so it is _cached in the environment_ when first calling the script. 
Afterwards, no recompilation is triggered when calling the solver globally.
The functions that use numba are denoted with the `_jit` suffix. They should work automatically. 
If you are changing the solver code in [solver.py](solver.py), the same changes should be applied to the [jit_solver_helpers.py](jit_solver_helpers.py).

### Progress Bar
If you wish to have a progress bar displayed during the integration you can use the `pbar` argument of the `integrate` function. 
This argument can either be `None` to indicate that no progress bar should be used or a [tqdm](https://github.com/tqdm/tqdm) pbar.
If the argument is a tqdm pbar it will be incremented by 1 for each timestep. An example usage is:
~~~python
from tqdm.auto import tqdm
T = (0,4000) # integrate from 0 to 4000
dt = 0.01 # timestep is 0.01
pbar = tqdm(total=int( (T[1] - T[0])/dt ))
t,x,v = slv.integrate(x_0,(0,4000),0.01,mat,(input,target),hist="both",pbar=pbar)
~~~

The `integrate_jit` function does **not** support progress bars!
## Simulation 

The simulation class handles simulating the evolutionary dynamics. It keeps track of the current strains (unique regulatory matrices) and gene expression states.
It can iterate to the next generation (integrate) and calculate the fitness scores. Furthermore, it can be used to create daughter strains from the most fit strain. 
It is design for easy extension. It is possible to implement new fitness scores or new selection criteria.

### Usage
We start by importing the simulation and creating an instance. We need to import numpy and multiprocessing too.

~~~python
from Simulation import Simulation
import multiprocessing as mp
import numpy as np


#Simulation Settings
args = {
    "dt":0.01,
    "generation_length":4000,
    "p1": 0.02, #these control the density of the random regulatory matrices
    "p2": 0.02,
    "epsilon":0.001 #you can also set model constants (the args dict is also passed to the solver instances)
}

if __name__ == "__main__": #required for multithreading

    #set the target genes
    target = np.ones(8)
    
    #set the input genes
    input = np.ones(6)
    
    sim = Simulation(args=args, environment=(input,target))

~~~

Then we need to add strains: 

~~~python
    sim.add_strain(N=20) #add a single strain (unique regulatory matrix) with 20 different starting points.
~~~
Or we can add 5 strains by:
~~~python
    for i in range(5):
        sim.add_strain(N=20)
~~~

If you have a specific regulatory matrix that you want to associate with the strain:
~~~python
    sim.add_strain(N=20,reg_mat=mat)
~~~

You can use print(sum) to print a list of included strains. Furthermore you can access the strains in the simulation with: 

~~~python
    for strain in sim.strains:
        print(strain.name)
        print(strain.reg_mat)
        print(strain.avg_v)

~~~
The names are assigned randomly. 

Next we want to  integrate the strains in our simulation a single generation. For this we need a multiprocessing pool: 
~~~python
    pool = mp.Pool(8)
    nice_pool  = mp.Pool(8, processes=10, maxtasksperchild=40, initializer=os.nice, initargs=(20,))
    sim.generation(pool)
~~~
This will integrate all registered strains by the `"generation_length"`. The different strains will be integrated in parallel.
We can calculate the fitness scores by: 

~~~python
    scores = sim.calculate_scores(pool,method="single_growth_rate")
~~~
we use `method="single_growth_rate"` to calculate the scores based on a single "generation". To simulate evolution based on these scores:
~~~python
    from Simulation import generate_prob_mat
    prob_mat = generate_prob_mat(0.007,0.007)
    sim.evolution(pool,method="best",scores=scores,probs=prob_mat)
~~~
The function generate_prob_mat is used to generate a transition probability matrix based on the p_d and p_c. For more details see [the definition](jit_solver_helpers.py). 

If we want to simulate two environments in sequence:

~~~python
    #environment 1
    target1 = np.ones(8)
    input1 = np.ones(6)
    
    #environment 2
    target2 = np.ones(8)
    input2 = np.ones(6)
    
    #simulate environment 1
    sim.environment = (input1,target1)
    sim.generation(pool)
    _ = sim.calculate_scores(pool,method="multi_growth_rate_mean",period=2)# we have two enviroments
    # calculate_Scores will return None since we should do the second environment before calling evoultion
    
    #simulate environment 2
    sim.environment = (input2,target2)
    sim.generation(pool)
    scores = sim.calculate_scores(pool,method="multi_growth_rate_mean",period=2)# we have two enviroments
    sim.evolution(pool,method="best",scores=scores,probs=prob_mat)    
 
~~~

The state of the strains are kept between the generations. However, calling `sim.evolution` resets the gene states (among other effects).
For a full example please refer to the scripts provided. 
### Arguments for the Simulator Class
The arguments are shown in the table below. These should be stored in a dict object and passed to the class instantiator as `args`. 


| argument                | default value | definition                                                                                                                                                                                                               |
|-------------------------|---------------|--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| arguments of the solver | -             | The arguments defined in the table above for the solver are also valid arguments for the simulation.                                                                                                                     |
| N_t                     | 8             | Number of target genes. **This value is hard coded in the [sim_helpers.py](Simulation/sim_helpers.py) `mutation` function. This should be remediated before using the `sim.evolution` function with a different value.** |
| N_i                     | 6             | Number of input genes                                                                                                                                                                                                    |
| p1                      | 0.02          | Used when generating random regulatory matrices. Is the probability for a +1 pathway                                                                                                                                     |
| p2                      | 0.02          | Used when generating random regulatory matrices. Is the probability for a -1 pathway                                                                                                                                     |
| dt                      | 0.01          | Time step for the simulation.                                                                                                                                                                                            |
| generation_length       | 4000          | How long a generation is. Default is 4000s.                                                                                                                                                                              |


### Extending the simulation
The easiest way to extend the simulation is to define new `method`'s for the `evolution` and `calculate_scores`.
By defining new methods, you can implement new fitness scores or new selection criteria. 
We have provided templates for defining new method's.
You can find them in the [simulation.py](simulation.py) within the relevant functions. 

You can also change how the mutation function works by modifying the `mutation` function in [sim_helpers.py](Simulation/sim_helpers.py).
The arguments of this function are  `probs` and `old_mat`. `probs`  is directly passed forward from the `sim.evolution` call. 
Currently, it contains the mutation probabilities, but it can be any arbitrary object. So, you can pass any arguments to this function.
`old_mat` is the parent regulatory matrix to be mutated. The function should return a matrix of the same size, containing the new regulatory matrix. 
### Notes
- `p1` and `p2` arguments control the probability of +1 or -1 pathways. Exactly speaking they define the probability of a pathway _that is not constrained by the model_ being +1 or -1. `1 -p1 -p2` is the probability of the pathway being 0. 
- The regulatory matrices have should have the shape `(N, N+N_i)`. The last` N_i` columns define the effects of the inputs. Since the inputs should be kept constant, the last `N_i` rows are non-necessary. 
- When used in multiple generation mode, the `calculate_scores` function should return `None`, if no scores can be calculated (not enough generations have passed). It should return scores, if enough generations have passed to calculate the fitness scores.
- Both extendable functions support arbitrary keyword arguments. When defining new methods, you can also use new arguments without changing the signature. Just use `kwargs["your_argument"]` to access the new argument.