#!/usr/bin/python
#/* -.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.
#* Purpose : Grant Proposal on Resource-Constrained Networks Simulation
#* Creation Date : 07-11-2017
#* Last Modified : Mon 13 Nov 2017 14:42:02 CST
#* Created By : Vasilis Livanos <basilis3761@yahoo.gr>
#_._._._._._._._._._._._._._._._._._._._._.*/

import cvxopt
import numpy as np
import igraph as ig
import random
import seaborn as sb
from cvxpy import *
import matplotlib.pyplot as plt
import sys

# Turn interactive mode off
plt.ioff()

# We assume that for a specific service provider
# the quality they provide for a specific type
# lies in [0, 1).

# Number of rounds
#T = 300
T = int(sys.argv[1])

# Number of agents
#N = 50
N = int(sys.argv[2])

# Resource budget
#beta = 100
beta = int(sys.argv[3])

# Number of service providers
#k = 10
k = int(sys.argv[4])

# Number of agent types
#Nq = 10
Nq = int(sys.argv[5])

# Graph generator function (number of agents, edge probability, preference matrix)
def generate_graph(n):

    # Instantiate the graph
    m = int(np.sqrt(n))
    graph = ig.Graph.Lattice([m, m], nei=1, circular=False)

    # Initialize graph, weights, proposals and provider qualities per type
    w = [[0 for x in range(n)] for y in range(n)]
    f = [[0 for x in range(n)] for y in range(n)]
    prov = [[random.random() for x in range(Nq)] for y in range(k)]
    for edge in graph.es:
        w[edge.source][edge.target] = random.random()
        w[edge.target][edge.source] = random.random()
        f[edge.source][edge.target] = random.uniform(0, beta)
        f[edge.target][edge.source] = random.uniform(0, beta)

    # Assign each agent a type, a provider, the provider's quality,
    # initialize the total quality, compute the set of neighbors
    # of the same type and initialize their beta distribution
    for i in range(n):
        graph.vs[i]['type'] = random.randint(0, Nq-1)
        graph.vs[i]['provider'] = random.randint(0, k-1)
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
    curr_max = 0
    #print("In best-response for i: ", i)
    solution = []
    for dlt in range(beta):
        delta = 0
        x = Variable(N)
        l = []
        for j in range(N):
            if(w[i][j] != 0):
                l.append(-w[i][j]*square(x[j]) + w[i][j]*beta*x[j])
        objective = Maximize( sum(l) ) 
        constraints = []
        for j in range(N):
            constraints += [0 <= x[j], x[j] <= f[j][i] ]
        constraints += [sum(x) <= beta]
        if(len(g.vs[i]['Z']) > 0):
            delta = dlt
            constraints += [sum(list(x[j] for j in g.vs[i]['Z'])) == delta]
        prob = Problem(objective, constraints)
        if(not(prob.is_dcp())):
            print("Not DCP.")
        val = prob.solve() + (1 - g.vs[i]['quality'])*1.0*(c*delta)/(1+c*delta) 
        if(val > curr_max):
            curr_max = val
            solution = x.value

    if(solution == []):
        return x.value
    else:
        return solution

def solver(graph, N, beta, k, Nq, p, w, f, opt, quality):

    qualities = [quality]
    
    # Main for-loop
    for t in range(T):

        # Pick agent at random
        i = random.randint(0, N-1)

        # Announce agent
        #print("Agent ", i, " updates their proposals.") 

        # Expected number of people that told agent i their quality
        samples = 0
        for j in graph.vs[i]['Z']:
            f_star = f[i][j]
            if(f[j][i] < f[i][j]):
                f_star = f[j][i]
            samples += 1.0*p*f_star
        samples = int(np.round(samples))

        # Select agents whose quality i learned and update i's quality
        #print("Number of agents of the same type: ", len(graph.vs[i]['Z']))
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
                #print("Agent updated their quality.")

        # Update the beta distribution and set theta to the mode
        if samples > 0:
            graph.vs[i]['a'] += samples - 1
            graph.vs[i]['b'] += 1
        if(not(graph.vs[i]['a'] == 1 and graph.vs[i]['b'] == 1)):
            graph.vs[i]['theta'] = 1.0 * (graph.vs[i]['a'] - 1) / (graph.vs[i]['a'] + graph.vs[i]['b'] - 2)
        else:
            graph.vs[i]['theta'] = 0.5

        f[i] = best_response(graph, w, f, i)

        # Append new quality point
        #print(sum(graph.vs[j]['quality'] for j in range(N)))
        qualities.append(sum(graph.vs[j]['quality'] for j in range(N)))

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

# Probability of passive information diffusion
p = 0.005
print("Data for p = ", p)
q1 = solver(g, N, beta, k, Nq, p, w, f, opt, init_qual)
if(q1 == None):
    print("q1 is None.")
p = 0.05
print("")
print("Data for p = ", p)
q2 = solver(g, N, beta, k, Nq, p, w, f, opt, init_qual)
if(q2 == None):
    print("q2 is None.")
p = 0.1
print("")
print("Data for p = ", p)
q3 = solver(g, N, beta, k, Nq, p, w, f, opt, init_qual)
if(q3 == None):
    print("q3 is None.")

fig, ax = sb.mpl.pyplot.subplots(1, 1)
colors = sb.mpl_palette('magma', n_colors=3)

ax.plot(
        range(T),
        q1,
        label=r'$p=0.005$',
        color=colors[0],
        linestyle='solid',
        linewidth=1)
ax.plot(
        range(T),
        q2,
        label=r'$p=0.05$',
        color=colors[1],
        linestyle='dotted',
        linewidth=1)
ax.plot(
        range(T),
        q3,
        label=r'$p=0.1$',
        color=colors[2],
        linestyle='dashed',
        linewidth=1)
z = ax
z.set_ylabel(r"Quality / OPT")
z.set_xlabel(r"Time")
z.legend(fontsize=6)
z.set_xlim([0, T])
z.set_xticks(np.arange(0, T, 1.0 * T / 10), minor=True)
z.set_ylim([0.3, 1])
z.set_yticks(np.arange(0.3, 1, 0.05), minor=True)
z.tick_params(direction='in', length=6, width=2, colors='k', which='major', labelsize=8)
z.tick_params(direction='in', length=4, width=1, colors='gray', which='minor')
#z.set_aspect(aspect='auto', adjustable='box-forced')
#z.xaxis.set_visible(False)
z.minorticks_on()

s = "./Grid_T=%s_N=%s_b=%s_k=%s_Nq=%s.pdf" % (T, N, beta, k, Nq)
sb.despine(offset=25, left=True, trim=True)
fig.tight_layout()
fig.savefig(s)
