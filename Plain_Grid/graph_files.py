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

# Grid name
# file_prefix = int(sys.argv[1])
file_prefix = "Plain_Grid"

#Nq = int(sys.argv[1])
Nq = "10"

fig, ax = sb.mpl.pyplot.subplots(1, 1)
colors = sb.mpl_palette('magma', n_colors=3)

i = 0
T = 0
data_dic = {}

for filename in os.listdir("./"):
    if filename.beginsWith(file_prefix):
        content = []
        with open(filename) as f:
            content = f.readlines()

        if len(content) > 100:
            grid_param_name = content[1]
            grid_param_name_list = grid_param_name.split()
            if grid_param_name_list[14] == Nq:
                if grid_param_name not in data_dic:
                    data_dic[grid_param_name] = []
                content = content[3:]
                content = [int(x.strip().split()[1]) for x in content]
                data_dic[grid_param_name].append(content)
        continue
    else:
        continue

# Average results and plot on graph by key
for key in data_dic.keys():
    number_of_tests = len(data_dic[key])
    average_q = [sum(x)/number_of_tests for x in zip(*data_dic[key])]
    T = len(average_q)

    ax.plot(
            range(T),
            average_q,
            label=key,
            color=colors[i],
            linestyle='solid',
            linewidth=1)
    i = i + 1

z = ax
z.set_ylabel(r"Average Quality / Graph Type for " + file_prefix)
z.set_xlabel(r"Time")
z.legend(fontsize=6)
z.set_xlim([0, T])
z.set_xticks(np.arange(0, T, 1.0 * T / 10), minor=True)
z.set_ylim([0.3, 1])
z.set_yticks(np.arange(0.3, 1, 0.05), minor=True)
z.tick_params(direction='in', length=6, width=2, colors='k', which='major', labelsize=8)
z.tick_params(direction='in', length=4, width=1, colors='gray', which='minor')
z.minorticks_on()

s = "./Grids_%s.pdf" % (name)
sb.despine(offset=25, left=True, trim=True)
fig.tight_layout()
fig.savefig(s)
