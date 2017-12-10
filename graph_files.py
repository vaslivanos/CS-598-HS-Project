#!/usr/bin/python
#/* -.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.

import cvxopt
import numpy as np
import igraph as ig
import random
import seaborn as sb
from cvxpy import *
import matplotlib.pyplot as plt
import sys

# graph name (for file naming)
graph_name = int(sys.argv[1])

#Probability
p = int(sys.argv[2])

#Folder and file names to read from
file_prefixs = ["Plain_Grid", "Biased_Grid", "Unbiased_grid"]

# the probability value used (for graph naming)
graph_type = "p=" + p

fig, ax = sb.mpl.pyplot.subplots(1, 1)
colors = sb.mpl_palette('magma', n_colors=3)

i = 0
T = 0
for file_prefix in file_prefixs:
    data_matrix = []
    for filename in os.listdir("./" + file_prefix):
        if filename.beginsWith(file_prefix):
            content = []
            with open(filename) as f:
                content = f.readlines()

            content = [int(x.strip().split()[1]) for x in content]
            data_matrix.append(content)
            continue
        else:
            continue

    number_of_tests = len(data_matrix)
    average_q = [sum(x)/number_of_tests for x in zip(*data_matrix)]
    
    if T == 0:
     T = len(average_data)

    ax.plot(
            range(T),
            average_q,
            label=file_prefix,
            color=colors[i],
            linestyle='solid',
            linewidth=1)
    i = i + 1

z = ax
z.set_ylabel(r"Average Quality / Graph Type for " + graph_type)
z.set_xlabel(r"Time")
z.legend(fontsize=6)
z.set_xlim([0, T])
z.set_xticks(np.arange(0, T, 1.0 * T / 10), minor=True)
z.set_ylim([0.3, 1])
z.set_yticks(np.arange(0.3, 1, 0.05), minor=True)
z.tick_params(direction='in', length=6, width=2, colors='k', which='major', labelsize=8)
z.tick_params(direction='in', length=4, width=1, colors='gray', which='minor')
z.minorticks_on()

s = "./Grids_%s.pdf" % (graph_name)
sb.despine(offset=25, left=True, trim=True)
fig.tight_layout()
fig.savefig(s)
