import argparse
import collections
import random
import sys, pdb
import site, os
from os import path

import numpy as np
import igraph as ig
from joblib import Parallel, delayed
import multiprocessing, time
from multiprocessing import Manager

import matplotlib
import matplotlib.pyplot as plt
import itertools
from mpmath import *

TINY_SIZE = 10
SMALL_SIZE = 12
MEDIUM_SIZE = 16
BIGGER_SIZE = 18

plt.rc('font', size=MEDIUM_SIZE)
plt.rc('axes', titlesize=SMALL_SIZE)
plt.rc('axes', labelsize=MEDIUM_SIZE)
plt.rc('xtick', labelsize=SMALL_SIZE)
plt.rc('ytick', labelsize=SMALL_SIZE)
plt.rc('legend', fontsize=TINY_SIZE)
plt.rc('figure', titlesize=BIGGER_SIZE)

def create_network(mean_degree, num_nodes):
    degree_sequence = np.random.poisson(mean_degree, num_nodes)
    while (np.sum(degree_sequence) % 2 !=0):
        degree_sequence = np.random.poisson(mean_degree, num_nodes)

    return ig.Graph.Degree_Sequence(list(degree_sequence))

def powerlaw_cutoff(a, b, kmax, n):
    normalization = polylog(a,np.exp(-1.0/b))
    probs = [0.0 for i in range(kmax)]
    for i in range(kmax-1):
        if i == 0:
            probs[i] = 0
        else:
            probs[i] = np.power(i,-1.0*a) * np.exp(-1.0*i/b) / normalization
    probs[kmax-1] = max(0,1 - sum(probs))
    elements = [i for i in range(kmax)]
    return np.random.choice(elements,n,p=probs)


def create_network_pl_cutoff(a,b,num_nodes):
    degree_sequence = powerlaw_cutoff(a,b,kmax=100,n=num_nodes)
    while (np.sum(degree_sequence) % 2 !=0):
        degree_sequence = powerlaw_cutoff(a,b,kmax=100,n=num_nodes)

    return ig.Graph.Degree_Sequence(list(degree_sequence))

def runExp(i, mean_degree1, mean_degree2,num_nodes, T1, mu1, T2, mu2):
    print('now running experiment ' + str(i))
    G1 = create_network(mean_degree1, num_nodes)
    G2 = create_network(mean_degree2, num_nodes)
    finalFrac = evolution(G1, G2, T1, mu1, T2, mu2)
    np.save('experiment_files/final_fractions_' + str(i), finalFrac)

def runExp_pl_cutoff(i, a1, b1, a2, b2, num_nodes, T1, mu1, T2, mu2):
    print('now running experiment ' + str(i))
    G1 = create_network_pl_cutoff(a1, b1, num_nodes)
    G2 = create_network_pl_cutoff(a2, b2, num_nodes)
    finalFrac = evolution(G1, G2, T1, mu1, T2, mu2)
    np.save('experiment_files/final_fractions_' + str(i), finalFrac)




def evolution(g1, g2, t1, mu1, t2, mu2):
    g1.simplify()
    g2.simplify()
    node_set = set(g1.vs.indices)
    num_nodes = len(node_set)
    # selecting a random patient zero
    patient_zero = int(np.random.randint(0, num_nodes - 1))
    infected = set([patient_zero])

    # setting up necessary dictionaries to track infection
    susceptible_nodes = node_set
    susceptible_nodes = susceptible_nodes.difference(infected)
    strain_list = [0 for i in range(num_nodes)]
    strain_list[patient_zero] = start_strain

    # converting probabilities into exponential rates
    r1 = [0.0, 0.0] # rates for layer 1
    r2 = [0.0, 0.0] # rates for layer 2
    r1[0] = np.log(1/(1-t1[0]))
    r1[1] = np.log(1/(1-t1[1]))
    r2[0] = np.log(1/(1-t2[0]))
    r2[1] = np.log(1/(1-t2[1]))

    # constructing neighbor-of-infected dictionary
    neighbors_of_infected_1 = dict()
    neighbors_of_infected_2 = dict()
    for node in g1.neighbors(patient_zero):
        expMean = r1[strain_list[patient_zero] -1]
        if expMean == 0:
            wt = 1
        else:
            wt = np.random.exponential(1.0 / expMean)
        if wt < 1:  # only add to dictionary if wt < 1, else patient zero recovers before infection
            neighbors_of_infected_1[node] = set([(patient_zero, wt)])
    for node in g2.neighbors(patient_zero):
        expMean = r2[strain_list[patient_zero] - 1]
        if expMean == 0:
            wt = 1
        else:
            wt = np.random.exponential(1.0/expMean)
        if wt < 1:
            neighbors_of_infected_2[node] = set([(patient_zero, wt)])

    while len(susceptible_nodes) > 0:

        # step 1: figuring out who gets infected next
        minE = None
        nextInfected = None
        nextHost = None
        edge_type = None
        for node in neighbors_of_infected_1:
            for val in neighbors_of_infected_1[node]:
                host, wt = val
                if (minE is None) or (minE < wt):
                    minE = wt
                    nextInfected = node
                    nextHost = host
                    edge_type = 1
        for node in neighbors_of_infected_2:
            for val in neighbors_of_infected_2[node]:
                host, wt = val
                if (minE is None) or (minE < wt):
                    minE = wt
                    nextInfected = node
                    nextHost = host
                    edge_type = 2
        if minE is None: break # no more infections happen
        # updating sets/dictionaries
        susceptible_nodes.remove(nextInfected)
        if nextInfected in neighbors_of_infected_1: neighbors_of_infected_1.pop(nextInfected)
        if nextInfected in neighbors_of_infected_2: neighbors_of_infected_2.pop(nextInfected)
        # determining mutation
        if edge_type == 1 and strain_list[nextHost] == 1:
            coinflip = np.random.binomial(1,mu1[0])
        elif edge_type == 1 and strain_list[nextHost] == 2:
            coinflip = np.random.binomial(1, 1.0-mu1[1])
        elif edge_type == 2 and strain_list[nextHost] == 1:
            coinflip = np.random.binomial(1, mu2[0])
        else:
            coinflip = np.random.binomial(1, 1.0-mu2[1])
        if coinflip == 1: # mutating to strain 1
            strain_list[nextInfected] = 1
        else:
            strain_list[nextInfected] = 2

        # generating new entries for neighbors_of_infected
        for node in g1.neighbors(nextInfected):
            if node not in susceptible_nodes: continue
            expMean = r1[strain_list[nextInfected] - 1]
            if expMean == 0:
                wt = 1
            else:
                wt = np.random.exponential(1.0/expMean)
            if wt >= 1.0: continue
            if node not in neighbors_of_infected_1:
                neighbors_of_infected_1[node] = set([(nextInfected, wt)])
            else:
                neighbors_of_infected_1[node].add((nextInfected, wt))

        for node in g2.neighbors(nextInfected):
            if node not in susceptible_nodes: continue
            expMean = r2[strain_list[nextInfected] - 1]
            if expMean == 0:
                wt = 1
            else:
                wt = np.random.exponential(1.0/expMean)
            if wt >= 1.0: continue
            if node not in neighbors_of_infected_2:
                neighbors_of_infected_2[node] = set([(nextInfected, wt)])
            else:
                neighbors_of_infected_2[node].add((nextInfected, wt))

    # computing final numbers
    num_strain_1 = 0
    num_strain_2 = 0
    for i in range(num_nodes):
        if strain_list[i] == 1: num_strain_1 += 1
        elif strain_list[i] == 2: num_strain_2 += 1

    return float(num_strain_1)/num_nodes, float(num_strain_2)/num_nodes

###############################################################################################

# SETTING GLOBAL PARAMETERS AND RUNNING ANALYSIS

###############################################################################################

# setting transmissibilities for g1 and g2
t1 = [0.6, 0.8] # for g1. t1[0] is the strain-1 transmissibility in g1.
t2 = [0.7, 0.9] # for g2

# setting mutation probabilities
mu1 = [0.1, 0.95] # for g1. mu1[0] is the strain 1-strain 1 mutation prob in g1, mu1[1] is the strain2-strain2 prob
mu2 = [0.1, 0.95] # for g2

# network parameters
num_nodes = 10000
mean_degree1 = 1
mean_degree2 = 1

# power law with exp cutoff
a1 = 2.5
b1 = 4
a2 = 2.3
b2 = 4

# simulation parameters
numExp = 1000
start_strain = 1
num_cores = 3
thrVal = 0.05 # threshold for determining if a giant component has emerged


###############################################################################################

# RUNNING SIMULATION

###############################################################################################

#Parallel(n_jobs=num_cores)(delayed(runExp_pl_cutoff)(j, a1, b1, a2, b2, num_nodes, t1, mu1, t2, mu2) for j in range(numExp))
#Parallel(n_jobs=num_cores)(delayed(runExp)(j, mean_degree1, mean_degree2, num_nodes, t1, mu1, t2, mu2) for j in range(numExp))

for j in range(numExp):
    # runExp_pl_cutoff(j,a1,b1,a2,b2,num_nodes,t1,mu1,t2,mu2)
    runExp(j,mean_degree1, mean_degree2, num_nodes, t1, mu1, t2, mu2)

total_final_frac = np.array([0.0,0.0])
num_epidemics = 0
# copying data to arrays
for i in range(numExp):
    cumulativeInfections = np.load('experiment_files/final_fractions_' + str(i) + '.npy')
    #print(cumulativeInfections)
    if sum(cumulativeInfections) > thrVal:
        total_final_frac = total_final_frac + cumulativeInfections
        num_epidemics += 1
if num_epidemics == 0: avg_final_frac = np.array([0.0, 0.0])
else:
    avg_final_frac = total_final_frac/num_epidemics

pe = float(num_epidemics)/numExp

print('Average fraction of strain 1 is ' + str(avg_final_frac[0]))
print('Average fraction of strain 2 is ' + str(avg_final_frac[1]))
print('Probability of emergence is ' + str(pe))

