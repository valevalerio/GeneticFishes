#!/usr/bin/env python
# coding: utf-8

# ## The Genetic Algorithm Class
# It's important to provide the following to the costructor:
# - fit_function: the function on which the chromosomes will be evaluated
# - pop_size: number of chromosomes
# - n_variables: how many variables are implicitly represented in the chromosome
# - bits: list of bits that will represent each variable.
# - ranges: the boundaries for each variable
# 
# And for the methods:
# - $p_c$: probability of crossover, argument of crossover
# - $p_m$: probability of mutation, argument of mutation




import numpy as np
from mpl_toolkits.mplot3d import Axes3D

import matplotlib.pyplot as plt

def vec_in_range(vector,bounds,m):
    a = bounds[0]
    b = bounds[1]
    v = a + (b-a)*vector/((2**m)-1)
    return v

def base2_base10(vector):
    vector_10 = 0
    m = len(vector)
    for i in range(0,m):
        vector_10 = vector_10 + vector[i]*2**i 
    return vector_10

def int_representation(value,bounds,m):
    a = bounds[0]
    b = bounds[1]
    return (value-a)*((2**m) -1)/(b-a)
    
def base10_base2(number,m):
    number = int(number)
    return [(int(el) if el != ' ' else 0) for el in list(("{0:"+str(m)+"b}").format(number))][::-1]
    
class InvalidFun(Exception):
    pass

class GeneticAlgorithm():

    def __init__(self,fit_fun,pop_size=50,n_variables=2,bits=[18,15],ranges=[[-3,12.11],[4.5,5.8]]):
        assert n_variables==len(bits),'precision has'+len(bits)+'elements but variables are'+n_variables
        assert np.unique([len(el) for el in ranges])==[2],'Not all the ranges are correct'
        
        self.pop_size=pop_size
        
        self.n_vars = n_variables
        self.intervals = ranges
        self.bits = bits
        self.population = self.init_population()
        print('population initialized\nChecking Function...',end=' ')
        self.fitness_function = fit_fun
        '''try:
            chrom=self.population[0]
            idxs = [0]+self.bits
            values=[]
            for j in range(self.n_vars):
                var_bin = chrom[idxs[j]:idxs[j]+idxs[j+1]]
                value = np.concatenate((values,var_bin),axis=0)
                
                values.append(
                    vec_in_range(
                        vector=base2_base10(var_bin),
                        bounds = self.intervals[j],
                        m=idxs[j+1]
                ))
            
            self.fitness_function(np.array(values))
            print('Correct')
        except:
            pass
            raise InvalidFun('fitness Function cannot be applied')'''
        
        
        
    def init_population(self,values=None):
        pop_size=self.pop_size
        chrom_population = []
        if values is None:
        
            for i in range(pop_size):
                chromosome=None
                for precision in self.bits:
                    ch_i = np.random.randint(0,2,precision)#variable i-th
                    chromosome = np.hstack([chromosome,ch_i])
                #print(chromosome)
                chrom_population.append(chromosome[1:])
        else:
            for v in values:
                chrom_population.append(
                    self.genetic_space_value(v)
                )
        return np.array(chrom_population)
    def genetic_space_value(self,values):
        #idxs = [0]+[np.sum(self.bits[:j+1]) for j in range(len(self.bits))]
        chrom = []
        for j in range(self.n_vars):
            
            
            chrom = chrom+base10_base2(
                int_representation(values[j],self.intervals[j],self.bits[j]),#like 42,
                self.bits[j]
            )
        return np.array(chrom)
            
    def original_space_value(self,index_best):
        idxs = [0]+[np.sum(self.bits[:j+1]) for j in range(len(self.bits))]
        chrom=self.population[index_best]
        value = None
        values = []
        for j in range(self.n_vars):
            var_bin = chrom[idxs[j]:idxs[j+1]]
            
            #value = np.concatenate((values,var_bin),axis=0)
            
            v = vec_in_range(
                    vector=base2_base10(var_bin),
                    bounds = self.intervals[j],
                    m=self.bits[j]
            )
            values.append(v)
        return np.array(values)
    def original_space_speace(self):
        original_values = []
        for i,chrom in enumerate(self.population):
            original_values.append(self.original_space_value(i))
        return np.array(original_values)
    
    def evaluation(self):
        #evaluate population
        fitness_values = []
        #print('evaluating',len(population),'individuals')
        idxs = [0]+[np.sum(self.bits[:j+1]) for j in range(len(self.bits))]
        for i,chrom in enumerate(self.population):
            value = None
            values = []
            for j in range(self.n_vars):
                var_bin = chrom[idxs[j]:idxs[j+1]]
                
                #value = np.concatenate((values,var_bin),axis=0)
                values.append(
                    vec_in_range(
                        vector=base2_base10(var_bin),
                        bounds = self.intervals[j],
                        m=self.bits[j]
                ))
            #    print('var',j,'=',values[-1],'represented by',var_bin)
            fitness_values.append(self.fitness_function(np.array(values)))
        self.fitness_values = fitness_values
        return fitness_values

    def best_chromosome(self):

        #fitness_values = evaluation(population)
        
        idx_best_one = np.argsort(self.fitness_values)[-1]
        fit_val = self.fitness_values[idx_best_one]
                    
        best_chrom = self.population[idx_best_one]
        values = []
        idxs = [0]+[np.sum(self.bits[:j+1]) for j in range(len(self.bits))]
        for j in range(self.n_vars):
            var_bin = best_chrom[idxs[j]:idxs[j+1]]
            value = np.concatenate((values,var_bin),axis=0)
            values.append(
                vec_in_range(
                    vector=base2_base10(var_bin),
                    bounds = self.intervals[j],
                    m=self.bits[j]
                )
            )

        return (fit_val,values)

    def roulette_wheel(self):
        self.population
        self.pop_size
        self.fitness_values

        new_population = []
        #normalize in [0, max+min]
        minimum = np.min(self.fitness_values)
        #print('GA: minim fitness is',minimum)
        #normalize in [0 .. max+minimum]
        if minimum>0:
            minimum=np.array([0])
        F_total = sum(self.fitness_values-minimum)
        prob_selection = (self.fitness_values-minimum)/F_total
        q = np.cumsum(prob_selection)
        for t in range(0,self.pop_size):
            r = np.random.rand(1)
            
            idx_chrom = min(np.argwhere(r < q))[0]
            new_population.append(self.population[idx_chrom])
            
        self.population = np.array(new_population)
        return(np.array(new_population))

    def crossover(self,p_c):
        

        r = np.random.rand(self.pop_size)
        cross_population = self.population[(r< p_c)]
        idx_cross = np.argsort(r<p_c)

        if len(cross_population)%2 != 0:
            cross_population = cross_population[:-1]


        for i in range(0,np.int(len(cross_population)/2)):

            pos = np.int(np.random.randint(1,np.sum(self.bits),1))

            bi = cross_population[2*i]
            bi_prima = bi
            bi_1 = cross_population[2*i+1]
            bi_1_prima = bi_1

            bi_1_prima[pos:] = bi[pos:]
            bi_prima[pos:] = bi_1[pos:]

            self.population[idx_cross[2*i]] = bi_prima
            self.population[idx_cross[2*i+1]] = bi_1_prima
        return self.population

    def mutation(self,p_m):
        
        for chrom in self.population:
            for j in range(0,len(chrom)):
                r = np.random.rand(1)[0]
                if (r < p_m):
                    chrom[j] = chrom[j]^1 #complement
        return self.population

        

def new_crossover(self,p_c):
        r = np.random.rand(self.pop_size)
        cross_population = self.population[(r< p_c)]
        idx_cross = np.argsort(r<p_c)

        if len(cross_population)%2 != 0:
            cross_population = cross_population[:-1]


        for i in range(0,np.int(len(cross_population)/2)):
            #I will instead choose a 2 randint
            #the first to choose how many neurons switch,
            #the second to decide which ones
            n_neurons = np.random.randint(1,3,1) #1 is just 1 up to all neurons-1 (3-1)
            neurons = np.random.choice([i for i in range(3)],n_neurons,#up to the fourth neuron
                                          replace=False)
            
            bi = cross_population[2*i]
            bi_prima = bi
            bi_1 = cross_population[2*i+1]
            bi_1_prima = bi_1
            for k in neurons:
                pos = 3*5*k #here switch the neuron in the rapresent.
                pos_1 = 3*5*(k+1)
                bi_1_prima[pos:pos_1] = bi[pos:pos_1]
                bi_prima[pos:pos_1] = bi_1[pos:pos_1]

            self.population[idx_cross[2*i]] = bi_prima
            self.population[idx_cross[2*i+1]] = bi_1_prima
        return self.population

