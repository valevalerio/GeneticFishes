import fishes
import matplotlib.pyplot as plt
import time
from fishes import run_simulation,refPt,np,rand,pi
from fish import Fish,initial_energy
from alivefood import AliveFood
import genalg 
from genalg import *
from gasupport import *#new_pop,plot_fitness,update_best_fish_file,stack_stat,new_food_pop,update_best_food_file,load_minds_from_file

import json



_,first_fish,first_food = run_simulation(fishes=None,max_time=1,n_pop=1,verbose=False,ALIVE=True)


###### 
###### 
###### FISH
###### 
###### 
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
n_pop_fish  = 30
# ### The genetic algorithm
# Initialize the enviroment
# population is initialized in costructor of the genetic Algorithm

ga_fish = GeneticAlgorithm(fit_fun=None,
                      bits=bits,
                      n_variables=n_variables,
                      pop_size=n_pop_fish,
                      ranges=ranges)



minds,colors = load_minds_from_file(namefile='framesf_f/best_food_eaters.json',pop_size=n_pop_fish)
ga_fish.init_population(minds)
#istanciate the Fish objects starting from the "minds"
fish_pop = new_fish_pop(minds,colors)


#stat for the fishes
energy = np.zeros(5)
eat= np.zeros(5)
all_div_speed = np.zeros(5)
all_div_turn = np.zeros(5)
fitness_values = np.zeros(5)
foods = np.zeros(5)

###### 
###### 
###### FOOD
###### 
###### 
n_variables = len(first_food[0].mind)
print(n_variables,first_food[0].mind)
input()
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
n_pop_food  = 30
# ### The genetic algorithm
# Initialize the enviroment
# population is initialized in costructor of the genetic Algorithm

ga_food = GeneticAlgorithm(fit_fun=None,
                      bits=bits,
                      n_variables=n_variables,
                      pop_size=n_pop_food,
                      ranges=ranges)


minds = load_food_minds_from_file(namefile='framesf_f/best_foods.json',pop_size=n_pop_fish)
ga_food.init_population(minds)
#minds = ga_food.original_space_speace()
#istanciate the Fish objects starting from the "minds"
food_pop = new_food_pop(minds)




max_gen=120

#stat for the fishes
energy = np.zeros(5)
eat= np.zeros(5)
all_div_speed = np.zeros(5)
all_div_turn = np.zeros(5)
fitness_values = np.zeros(5)
foods = np.zeros(5)

#stat for the food
lifetimes = np.zeros(5)

folder = 'advanced_frames'

for i in range(max_gen):
    
    #Evaluate current population using the simulation
    print('Gen',str(i))
    start = time.time()
    if i %5 == 0 or i== max_gen-1:
        num_food,fg_i,food_generation_i = run_simulation(fishes=fish_pop,
                                                        foods=food_pop,
                                                        max_time=200,

                                                        n_pop=n_pop_fish,
                                                        epoch=i,
                                                        verbose=3,
                                                        record=False,
                                                        ALIVE=True,
                                                        folder=folder)
        end = time.time()
        print(round(end - start,1),'time needed for simulation shown')
    else:    
        num_food,fg_i,food_generation_i = run_simulation(fishes=fish_pop,
                                                        foods=food_pop,
                                                        max_time=200,
                                                        n_pop=n_pop_fish,
                                                        epoch=i,
                                                        verbose=False,
                                                        ALIVE=True,
                                                        folder=folder)
        end = time.time()
        print(round(end - start,1),'time needed without simulation shown')
    
    print(len(food_generation_i))
    print([f.lifetime for f in food_generation_i])

    #same as ga.evaluation
    energies,eaten,div_speed,div_turn = zip(*[(f.energy*(f.energy>0),f.eaten,f.diversity_speed,f.diversity_turn) for f in fg_i])
    div_speed = 10 * 1/(1+np.std(np.array(div_speed),axis=1))
    div_turn = 10 * 1/(1+np.std(np.array(div_turn),axis=1))
    eaten = np.array(eaten)
    energies = (np.array(energies)*eaten)
    

    ga_fish.fitness_values = energies+20*div_speed+20*div_turn + 10*eaten

    ga_food.fitness_values = np.array([f.lifetime for f in food_generation_i])


    #Find the best solution in current population
    max_energy,x_best = ga_fish.best_chromosome()
    #plotting in the meanwhile, the array of the stats got upgraded with the percentiles
    energy,eat,all_div_speed,all_div_turn,fitness_values,foods,lifetimes=plot_fitness(
                    [energy,    eat,    all_div_speed,  all_div_turn,  fitness_values,foods,lifetimes],
                    [energies,  eaten,  div_speed,      div_turn,      ga_fish.fitness_values, num_food,ga_food.fitness_values],
                    ['energy','eaten','diversity speed','diversity turn','fitness fun','food','food lifetime'],
                    figname=folder+'/curves_moving_food.png')
    
    
    update_best_fish_file(fg_i,generation=i,name=folder+'/best_food_eaters')
    update_best_food_file(food_generation_i,generation=i,name=folder+'/best_foods')
    #Select a new population
    ga_fish.roulette_wheel()
    ga_food.roulette_wheel()

    #Crossover operation
    ga_fish.crossover(p_c=0.3)
    ga_food.crossover(p_c=0.3)
    #Mutation operation
    ga_fish.mutation(p_m=0.015)
    ga_food.mutation(p_m=0.01)

    #update the new population for the simulation
    fish_pop = new_fish_pop(ga_fish.original_space_speace(),colors=[np.array(f.color) for f in fg_i])
    food_pop = new_food_pop(ga_food.original_space_speace())


