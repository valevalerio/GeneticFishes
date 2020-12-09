import fishes
import cv2
import matplotlib.pyplot as plt
import time
from fishes import run_simulation,refPt,np,rand,pi,math
from fish import Fish,initial_energy
import genalg 
from genalg import *
from gasupport import *
from alivefood import AliveFood
import json

def PointsInCircum(r,n=100):
    return np.array([(math.cos(2*pi/n*x)*r,math.sin(2*pi/n*x)*r) for x in range(0,n+1)],dtype=np.int)


size_w = (600,900,3)
size_w = size_w
size_w1,size_w2 = size_w[:2][::-1]
s1= 100
s2 = 100

food_file = "advanced_frames/best_foods.json" #the best surviving food

predators_file = "advanced_frames/best_food_eaters.json" #the best trained with moving energy
fish_file="./current5best_individuals.json" #trained fishes on food still

#next is to evolve the foods and the fishes again simultanuosly

with open(fish_file, "r") as write_file:
    data_fishes = json.load(write_file)

with open(predators_file, "r") as write_file:
    data_predator = json.load(write_file)

bests,colors = zip(*[(d['values'],np.array(d['color'])) for d in data_fishes])
print(colors)
num_best = len(bests)
fishes = [Fish(color=colors[i%num_best],mind=np.array(bests[i%num_best])) for i in range(15)]

bests,colors = zip(*[(d['values'],np.array(d['color'])) for d in data_predator])
print(colors)
num_best = len(bests)

predators = [Fish(color=colors[i%num_best],mind=np.array(bests[i%num_best])) for i in range(15)]
color=np.array((100,1,1))
f_s = fishes
print(len(f_s))
for f in f_s:
    f.pos= (rand.randint(s1,size_w2-s1),#x
                            rand.randint(s2,size_w1-s2))

with open(food_file, "r") as write_file:
    data = json.load(write_file)

preys = [d['values'] for d in data]
num_best = len(preys)
foods = [AliveFood(mind=np.array(preys[i%num_best])) for i in range(20)]

for f in foods:
    f.pos= (rand.randint(s1,size_w2-s1),#x
                            rand.randint(s2,size_w1-s2))

num_food,first_fish,first_foods = run_simulation(fishes=f_s,size_w=size_w,max_time=1000,epoch=19,
                            verbose=2,record=False,
                            foods=foods,
                            ALIVE=True,
                            folder='debugFrame')