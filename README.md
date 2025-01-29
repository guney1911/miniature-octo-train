# Simulating Noise-Driven Adaptation and Evolution
Güney Erin Tekin
contact: gtekin@student.ethz.ch guney.erin@gmail.com

This repository contains the code that we wrote during the Autumn Semester 2024, while the author was on exchange in the University of Tokyo.
The main goal of our research was to extend the model by Prof. Furusawa and Sato-san to include evolutionary dynamics.
For results and detailed explanations please refer to the [presentation slides](presentation.pdf). 

The repository contains the [Simulation](Simulation) package, which is self-contained package that has the capability to:
- Solve the model (integrate the model) using Euler-Maruyama-Method for multiple starting points simultaneously
- To run a simulation containing multiple strains (unique regulatory matrices) with multiple starting points for each strain
- To calculate fitness scores and do selection + mutation based on the calculated scores
- The simulation is designed to be easily extendable. Templates are provided for following extensions:
  - New fitness scores
  - New selection criteria

Please refer to the relevant documentation [here](Simulation/README.md). The package can be easily installed by the command:

~~~shell
pip install ./Simulation/
~~~
When changing the simulation code, please do not forget to reinstall the library after editing the code in the [Simulation](Simulation) directory.

There are example scripts for the stand-alone solver and the full simulation provided in the [Examples](Examples) directory.
Examples from the scripts we used for our research are provided in the [Old_Code](Old_Code) directory.
These scripts are much longer than the example scripts and contain the necessary code to run the scripts on the cluster and log the results.
Some of them are also untested with the current version of the library. Please read the example scripts first.

