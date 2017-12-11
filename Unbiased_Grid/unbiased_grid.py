#!/usr/bin/python
#/* -.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.
#* Purpose : Grant Proposal on Resource-Constrained Networks Simulation
#* Creation Date : 07-11-2017
#* Last Modified : Tue 14 Nov 2017 23:18:50 CST
#* Created By : Vasilis Livanos <basilis3761@yahoo.gr>
#_._._._._._._._._._._._._._._._._._._._._.*/

import cvxopt
import numpy as np
import igraph as ig
import random
from cvxpy import *
import sys

# We assume that for a specific service provider
# the quality they provide for a specific type
# lies in [0, 1).

# Number of rounds
T = int(sys.argv[1])

# Number of agents
N = int(sys.argv[2])

# Resource budget
beta = int(sys.argv[3])

# Number of service providers
k = int(sys.argv[4])

# Number of agent types
Nq = int(sys.argv[5])

# Probability p of inrmation diffusion
p = float(sys.argv[6])

# location and prefix for data files
FILENAME_PREFIX = "../data/data_" + str(N) + "_" + str(beta) + "_" + str(k) + "_" + str(Nq) + "_"

# Graph generator function (number of agents, edge probability, preference matrix)
def generate_graph(n):

    # Instantiate the graph
    m = int(np.sqrt(n))
    graph = ig.Graph.Lattice([m, m], nei=1, circular=False)

    dist = [[0 for i in range(n)] for j in range(n)]
    for i in range(n):
        for j in range(n):
            x = i / n
            y = i % n
            z = j / n
            w = i % n
            if(i != j):
                dist[i][j] = 1.0 / ((np.absolute(x-z) + np.absolute(y-w))**2)
        nrmlz = np.sum(dist[i])
        for j in range(n):
            dist[i][j] /= nrmlz
        l = range(n)
        l2 = dist[i]
        neighbor = np.random.choice(l, 1, p=l2)
        graph.add_edge(i, neighbor[0])

    # Initialize graph, weights, proposals and provider qualities per type
    w = [ list(map(float,x)) for x in read_csv_data(FILENAME_PREFIX + "w") ]
    f = [ list(map(float,x)) for x in read_csv_data(FILENAME_PREFIX + "f") ]
    prov = [ list(map(float,x)) for x in read_csv_data(FILENAME_PREFIX + "prov") ]
    types = list(map(int, read_csv_data(FILENAME_PREFIX + "types")[0]))
    providers = list(map(int, read_csv_data(FILENAME_PREFIX + "providers")[0]))


    # Assign each agent a type, a provider, the provider's quality,
    # initialize the total quality, compute the set of neighbors
    # of the same type and initialize their beta distribution
    for i in range(n):
        graph.vs[i]['type'] = types[i]
        graph.vs[i]['provider'] = providers[i]
        graph.vs[i]['quality'] = prov[graph.vs[i]['provider']][graph.vs[i]['type']]
        graph.vs[i]['Ns'] = set()
        graph.vs[i]['Z'] = set()
        graph.vs[i]['a'] = 1
        graph.vs[i]['b'] = 1

    for edge in graph.es:
        i,j = edge.source,edge.target
        graph.vs[i]['Ns'].add(j)
        graph.vs[j]['Ns'].add(i)
        if(graph.vs[i]['type'] == graph.vs[j]['type']):
            graph.vs[i]['Z'].add(j)
            graph.vs[j]['Z'].add(i)

    return (graph, w, f, prov)

# Best-response function (graph, weights, proposals, agent)
def best_response(g, w, f, i):
    # Calculate best-response
    c = p*(1 - g.vs[i]['theta'])
    x = Variable(N)
    l = []
    for j in range(N):
        l.append(-w[i][j]*square(x[j]) + w[i][j]*beta*x[j])
    objective = Maximize(sum(l))
    constraints = []
    for j in range(N):
        constraints += [0 <= x[j], x[j] <= f[j][i] ]
    constraints += [sum(x) <= beta]

    curr_max = 0
    solution = []
    for dlt in range(beta):
        delta = 0
        new_constraints = constraints
        if(len(g.vs[i]['Z']) > 0):
            delta = dlt
            new_constraints += [sum(list(x[j] for j in g.vs[i]['Z'])) == delta]
        prob = Problem(objective, new_constraints)
        if(not(prob.is_dcp())):
            print("Not DCP.")
        val = prob.solve() + (1 - g.vs[i]['quality'])*1.0*(c*dlt)/(1+c*dlt)
        if(val > curr_max):
            curr_max = val
            solution = x.value

    if(solution == []):
        return x.value
    else:
        return solution

def solver(graph, N, beta, k, Nq, w, f, opt, quality):

    qualities = [quality]

    # Main for-loop
    for t in range(T):

        # Pick agent at random
        i = random.randint(0, N-1)

        # Expected number of people that told agent i their quality
        samples = 0
        for j in graph.vs[i]['Z']:
            f_star = f[i][j]
            if(f[j][i] < f[i][j]):
                f_star = f[j][i]
            samples += 1.0*p*f_star
        samples = int(np.round(samples))

        # Select agents whose quality i learned and update i's quality
        if(samples >= len(graph.vs[i]['Z'])):
            s = set(graph.vs[i]['Z'])
        else:
            qs = []
            s = set()
            while len(s) < samples:
                smpl = random.choice(tuple(graph.vs[i]['Z']))
                qs.append(smpl)
                s = set(qs)
        qs = list(s)
        for ag in qs:
            if(graph.vs[i]['quality'] < graph.vs[ag]['quality']):
                graph.vs[i]['quality'] = graph.vs[ag]['quality']

        # Update the beta distribution and set theta to the mode
        if samples > 0:
            graph.vs[i]['a'] += samples - 1
            graph.vs[i]['b'] += 1
        if(not(graph.vs[i]['a'] == 1 and graph.vs[i]['b'] == 1)):
            graph.vs[i]['theta'] = 1.0 * (graph.vs[i]['a'] - 1) / (graph.vs[i]['a'] + graph.vs[i]['b'] - 2)
        else:
            graph.vs[i]['theta'] = 0.5

        solution = best_response(graph, w, f, i)
        if not solution is None:
            for k in range(len(solution)):
                f[i][k] = solution[k]

        # f[i] = best_response(graph, w, f, i)

        # Append new quality point
        qualities.append(np.sum(graph.vs[j]['quality'] for j in range(N)))

    ratio = []
    for i in range(T):
        ratio.append(1.0 * qualities[i] / opt)
        print(i, ratio[i])

    return ratio


(g, w, f, prov) = generate_graph(N)

init_qual = (np.sum(g.vs[i]['quality'] for i in range(N)))

# Calculate optimal quality
opt = 0
for i in range(N):
	maxq = prov[0][g.vs[i]['type']]
	for j in range(k):
		if(prov[j][g.vs[i]['type']] > maxq):
			maxq = prov[j][g.vs[i]['type']]
	opt += maxq

print("Unbiased Grid")
s = "T = %s N = %s beta = %s k = %s Nq = %s p = %s" % (T, N, beta, k, Nq, p)
print(s)
print("OPT = ", opt)
print("")

q = solver(g, N, beta, k, Nq, w, f, opt, init_qual)
if(q == None):
    print("q is None.")
