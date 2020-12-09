import fishes
import matplotlib.pyplot as plt
import time
from fishes import run_simulation,refPt,np,rand,pi
from fish import Fish,initial_energy
import genalg 
from genalg import *
from gasupport import new_pop,plot_fitness,update_best_fish_file,stack_stat

import json



_,first_fish = run_simulation(fishes=None,max_time=1,n_pop=1,verbose=False)
n_variables = len(first_fish[0].mind)

precision = 1 #precision digits after the decimal point
l_b = -1.1 #low_bound of each value
u_b = 1.1 #up bound of each value


n_bits = int(np.log2(np.abs(u_b-l_b)*10**precision))+1
print(n_bits,'bits for each variable')
print(n_bits*n_variables, 'total bits,chromosome length')


# ### The ranges
ranges = []
bits = []
for i in range(n_variables):
    #the ranges for each variable
    ranges.append([l_b,u_b])
    #precision for each variable (in bits)
    bits.append(n_bits)


n_pop  = 30
# ### The genetic algorithm
# Initialize the enviroment
# population is initialized in costructor of the genetic Algorithm

ga = GeneticAlgorithm(fit_fun=None,
                      bits=bits,
                      n_variables=n_variables,
                      pop_size=n_pop,
                      ranges=ranges)

minds = ga.original_space_speace()
#istanciate the Fish objects starting from the "minds"
fish_pop = new_pop(minds)

max_gen=120

energy = np.zeros(5)
eat= np.zeros(5)
all_div_speed = np.zeros(5)
all_div_turn = np.zeros(5)
fitness_values = np.zeros(5)
foods = np.zeros(5)

for i in range(max_gen):
    
    #Evaluate current population using the simulation
    print('Gen',str(i))
    start = time.time()
    if i in [0,1,10,31,50,101,150,max_gen-1]:
        food,fg_1 = run_simulation(fishes=fish_pop,max_time=200,n_pop=n_pop,epoch=i,verbose=True,record=True)
        end = time.time()
        print(round(end - start,1),'time needed for simulation shown')
    else:    
        food,fg_1 = run_simulation(fishes=fish_pop,max_time=200,n_pop=n_pop,epoch=i,verbose=False)
        end = time.time()
        print(round(end - start,1),'time needed without simulation shown')
    

    #same as ga.evaluation
    energies,eaten,div_speed,div_turn = zip(*[(f.energy*(f.energy>0),f.eaten,f.diversity_speed,f.diversity_turn) for f in fg_1])
    div_speed = 10 * 1/(1+np.std(np.array(div_speed),axis=1))
    div_turn = 10 * 1/(1+np.std(np.array(div_turn),axis=1))
    eaten = np.array(eaten)
    energies = (np.array(energies)*eaten)
    

    ga.fitness_values = energies+20*div_speed+20*div_turn + 10*eaten

    #Find the best solution in current population

    max_energy,x_best = ga.best_chromosome()
    #plotting in the meanwhile, the array of the stats got upgraded with the percentiles
    energy,eat,all_div_speed,all_div_turn,fitness_values=plot_fitness(
                    [energy,eat,all_div_speed,all_div_turn,fitness_values,foods],
                    [energies,eaten,div_speed,div_turn,ga.fitness_values,food],
                    ['energy','eaten','diversity speed','diversity turn','fitness fun','food'])
    min_foods = np.vstack([min_foods ,min_food])
    plt.plot(min_foods,label='minimum food')
    plt.savefig('min_food.png')
    plt.clf()
    plt.close()
    
    
    update_best_fish_file(fg_1)
    #Select a new population
    ga.roulette_wheel()

    #Crossover operation
    ga.crossover(p_c=0.3)

    #Mutation operation
    ga.mutation(p_m=0.015)

    #update the new population for the simulation
    fish_pop = new_pop(ga.original_space_speace())



