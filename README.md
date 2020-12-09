# GeneticFishes
## An old exercize. 
Used to apply a rough Genetic Algorithm Optimizer to train shallow NeuralNetworks inside each creature based on their goal. 
check here 

[![GenFISH](http://i3.ytimg.com/vi/DngpADQ740Y/hqdefault.jpg)](https://www.youtube.com/watch?v=DngpADQ740Y "GenFishes")
The fishes counsume energy to move and get energy from eating. 
The food just wants to survive :)
Those are the objective inducted by the fitness functions.

## Commands
In order to obtain similar and more accurate simulations as the one in the video is enougth to run: 

>python GAapplied.py 

will generate the weights for the neural networks of the fishes. it takes time for them to learn, standart epochs are 300, simulation time is 200 ticks.

>python python foodGAapplied.py 

will train the food to resist to a new swarm consisting on the best 5 fishes trained in the stage 1. 300 epochs, 200 ticks.
Also in the alive food there is a shallow NN, with different hyperparameters.

>./get_video.sh .

After each simulation is finished, is possible to obtain the video of the simulation (sampled each 20 epochs) of the creatures "learning" (frames of simulation will be deleted with this operation) 

>python debugfishes.py

can be used to check the check the behaviour of the neural networks of both the species. You can click on a fish and check what are its inputs and outputs.

OpenCV is used for the graphics.


