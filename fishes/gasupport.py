import numpy as np
import random as rand
from fish import Fish,pi
from alivefood import AliveFood
import json

import matplotlib.pyplot as plt
def new_fish_pop(minds,colors,size_w = (600,900,3),s1=100,s2=100):
    n_pop = minds.shape[0]
    size_w1,size_w2 = size_w[:2][::-1]
    pos_orient_color = [ (  rand.randint(s1,size_w2-s1),rand.randint(s2,size_w1-s2),2*pi*rand.random(),np.random.choice(range(256), size=3, replace=False)) for i in range(n_pop)]
    fishes = [Fish(orient=o,pos=(p1,p2),color=c) for p1,p2,o,c in pos_orient_color]
    for f,c,m in zip(fishes,colors,minds):
        f.color =  [np.asscalar(e) for e in c]
        f.mind = f.init_mind(variables = m)
    return fishes

def new_food_pop(minds,size_w = (600,900,3),s1=100,s2=100,max_radius=40):
    n_pop = minds.shape[0]
    size_w1,size_w2 = size_w[:2][::-1]
    foods_pos_orient=np.array([(  rand.randint(s2+1+max_radius,size_w2-s2-1-max_radius),#0-900
                                rand.randint(s1+1+max_radius,size_w1-s1-1-max_radius),#0-600
                                2*pi*rand.random())
                                         for i in range(n_pop)])

    foods = [AliveFood(pos=(p1,p2),orient=o) for p1,p2,o in foods_pos_orient]
    for f,m in zip(foods,minds):
        
        f.mind = f.init_mind(variables = m)
    return foods

def plot_fitness(list_attributes,new_values,names_attributes,figname = 'curves.jpg'):
    
    plt.subplot(111)
    n_row = len(list_attributes)
    plt.figure(figsize=(20,10))
    fig, axs = plt.subplots(nrows=n_row, ncols=1, sharex=True)
    
    for i,(att,att2,n) in enumerate(zip(list_attributes,new_values,names_attributes)):
        ax = axs[i]
        att = stack_stat(att,att2)
        list_attributes[i]=att
        percentile_numbers = att.shape[1]
        ax.plot(att[:,0],color='gray')
        for j in range(1,percentile_numbers):
            if j==percentile_numbers-1:
                ax.plot(att[:,j],color='black')
            elif j==int(percentile_numbers/2):
                ax.plot(att[:,j],color='red')
            else:
                ax.plot(att[:,j],color='gray')
        ax.set_title(n)
    #plt.legend(fontsize=15)
    plt.savefig(figname)
    plt.clf()
    plt.close()
    return (list_attributes )
    
def update_best_fish_file(fishes,generation,num_best=5,name='best_individuals'):
    try:
        with open(name+".json", "r") as read_file:
            data = json.load(read_file)
    except json.decoder.JSONDecodeError:
        data=[]

    data.sort(key=lambda x: x['energy'])
    #print(data)
    best_f = sorted(fishes, key=lambda f: f.energy, reverse=True)
    #print('printing the energies',[e.energy*e.eaten for e in best_f])
    if len(data) < num_best:
        while (len(data)<num_best):
            for i in range(0,num_best-len(data)):
                data.append({
                    'values':list(best_f[i].mind),
                    'eaten':best_f[i].eaten,
                    'energy':best_f[i].energy,
                    'color':best_f[i].color,
                    'generation':generation
                })
    else:
        j=0
        for i in range(len(data)):
            #print(i,data[i]['energy'], best_f[j].eaten*best_f[j].energy)
            if data[i]['energy']*data[i]['eaten']< best_f[j].eaten*best_f[j].energy:
                data[i]={
                    'values':list(best_f[j].mind),
                    'eaten':best_f[j].eaten,
                    'energy':best_f[j].energy,
                    'color':best_f[j].color,
                    'generation':generation
                }
                j+=1
            elif data[i]['eaten']<best_f[j].eaten:
                data[i]={
                    'values':list(best_f[j].mind),
                    'eaten':best_f[j].eaten,
                    'energy':best_f[j].energy,
                    'color':best_f[j].color,
                    'generation':generations
                }
                j+=1
            else:
                break
    with open(name+".json", "w") as write_file:
        json.dump(data, write_file)
def update_best_food_file(foods,generation,num_best=5,name='best_individuals'):
    try:
        with open(name+".json", "r") as read_file:
            data = json.load(read_file)
    except json.decoder.JSONDecodeError:
        data=[]

    data.sort(key=lambda x: x['lifetime'])
    #print(data)
    best_f = sorted(foods, key=lambda f: f.lifetime, reverse=True)
    #print('printing the energies',[e.energy*e.eaten for e in best_f])
    if len(data) < num_best:
        for i in range(0,num_best-len(data)):
            data.append({
                    'values':list(best_f[i].mind),
                    'generation':generation,
                    'lifetime':best_f[i].lifetime
                })
    else:
        j=0
        for i in range(len(data)):
            #print(i,data[i]['energy'], best_f[j].eaten*best_f[j].energy)
            if data[i]['lifetime']< best_f[j].lifetime:
                data[i]={
                    'values':list(best_f[i].mind),
                    'generation':generation,
                    'lifetime':best_f[i].lifetime
                }
                j+=1
            else:
                break
            
    

    with open(name+".json", "w") as write_file:
        json.dump(data, write_file)

def stack_stat(arr,arr2):
    #print(arr2)
    return np.vstack([arr,
            np.percentile(arr2,[5,25,50,75,95])
                ])


def load_minds_from_file(namefile,pop_size):
    with open(namefile, "r") as write_file:
        data_predator = json.load(write_file)
    bests,colors = zip(*[(d['values'],np.array(d['color'])) for d in data_predator])
    num_best = len(bests)
    minds,colors = zip(*[(np.array(bests[i%num_best]),colors[i%num_best]) for i in range(pop_size)])
    return np.array( minds),colors

def load_food_minds_from_file(namefile,pop_size):
    with open(namefile, "r") as write_file:
        data_predator = json.load(write_file)
    bests = [d['values'] for d in data_predator]
    num_best = len(bests)
    minds = [np.array(bests[i%num_best]) for i in range(pop_size)]
    return np.array(minds)