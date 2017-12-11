#!/usr/bin/python
#/* -.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.
#* Purpose : Grant Proposal on Resource-Constrained Networks Simulation
#* Creation Date : 07-11-2017
#* Last Modified : Tue 14 Nov 2017 23:18:58 CST
#* Created By : Vasilis Livanos <basilis3761@yahoo.gr>
#_._._._._._._._._._._._._._._._._._._._._.*/

import cvxopt
import numpy as np
import igraph as ig
import random
from cvxpy import *
import sys
import csv

# We assume that for a specific service provider
# the quality they provide for a specific type
# lies in [0, 1).

# Number of agents
N = int(sys.argv[1])

# Resource budget
beta = int(sys.argv[2])

# Number of service providers
k = int(sys.argv[3])

# Number of agent types
Nq = int(sys.argv[4])


# Graph generator function (number of agents, edge probability, preference matrix)
def generate_graph(n):
    types = [0 for x in range(n)]
    providers = [0 for x in range(n)]
    for i in range(n):
        types[i] = random.randint(0, Nq-1)
        providers[i] = random.randint(0, k-1)

    # Initialize graph, weights, proposals and provider qualities per type
    w = [[0 for x in range(n)] for y in range(n)]
    f = [[0 for x in range(n)] for y in range(n)]
    prov = [[random.random() for x in range(Nq)] for y in range(k)]

    m = int(np.sqrt(n))
    graph = ig.Graph.Lattice([m, m], nei=1, circular=False)
    for edge in graph.es:
        w[edge.source][edge.target] = random.random()
        w[edge.target][edge.source] = random.random()
        f[edge.source][edge.target] = random.uniform(0, beta)
        f[edge.target][edge.source] = random.uniform(0, beta)

    return (types, w, f, prov, providers)

def write_matrix(dataList, name):
    with open(name + ".csv","w") as f:
        wr = csv.writer(f)
        wr.writerows(dataList)

def write_list(dataList, name):
    with open(name + ".csv","w") as f:
        wr = csv.writer(f)
        wr.writerow(dataList)

(types, w, f, prov, providers) = generate_graph(N)

filename_prefix = "./data/data_" + str(N) + "_" + str(beta) + "_" + str(k) + "_" + str(Nq) + "_"

# write the data structures to a file
write_matrix(w, filename_prefix + "w")
write_matrix(f, filename_prefix + "f")
write_matrix(prov, filename_prefix + "prov")
write_list(types, filename_prefix + "types")
write_list(providers, filename_prefix + "providers")
