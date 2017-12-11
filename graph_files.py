#!/usr/bin/python
#/* -.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.

import seaborn as sb
import numpy as np
import matplotlib.pyplot as plt
import sys
import os

# Number of agent types. Set to `all` to graph all agent types
Nq = str(sys.argv[1])

# Probability p of information diffusion. Set to `all` to graph all p types
p = str(sys.argv[2])

fig, ax = sb.mpl.pyplot.subplots(1, 1)

# Grid name to filter
file_prefix = "Combind"

# The endex of the configuration values
# eg. (T = 100 N = 64 beta = 100 k = 50 Nq = 1 p = 0.1)
NQ_INDEX = 14
P_INDEX = 17

T = 0
data_dic = {}

def should_filter_file(grid_params_list):
    if Nq != "all" and grid_params_list[NQ_INDEX] != Nq:
        return True
    if p != "all" and grid_params_list[P_INDEX] != p:
        return True
    return False

for root, dirs, files in os.walk("."):
    for folder in dirs:
        if "Grid" not in folder:
            continue
        for filename in os.listdir("./" + folder):
            if filename.startswith(folder):
                content = []
                with open("./" + folder + "/" + filename) as f:
                    content = f.readlines()

                # Get the grid param values
                grid_params = content[1]
                grid_params_list = grid_params.split()

                # If the file should not be filtered, pars and add it to the data dictionary
                if not should_filter_file(grid_params_list):
                    grid_params = folder + "_" + grid_params
                    if grid_params not in data_dic:
                        data_dic[grid_params] = []
                    content = content[4:]
                    content = [float(x.strip().split()[1]) for x in content]
                    data_dic[grid_params].append(content)
                continue
            else:
                continue

num_colors = len(data_dic.keys());
colors = sb.mpl_palette('magma', n_colors=num_colors)

# Average results and plot on graph by key
color_index = 0
for key in data_dic.keys():
    T = len(data_dic[key][0])
    number_of_tests = len(data_dic[key])
    average_q = [sum(x)/number_of_tests for x in zip(*data_dic[key])]
    ax.plot(
            range(T),
            average_q,
            label=key,
            color=colors[color_index],
            linestyle='solid',
            linewidth=1)
    color_index = color_index + 1

z = ax
z.set_ylabel(r"Average Quality for " + file_prefix)
z.set_xlabel(r"Time")
z.legend(fontsize=6)
z.set_xlim([0, T])
z.set_xticks(np.arange(0, T, 1.0 * T / 10), minor=True)
z.set_ylim([0.3, 1])
z.set_yticks(np.arange(0.3, 1, 0.05), minor=True)
z.tick_params(direction='in', length=6, width=2, colors='k', which='major', labelsize=8)
z.tick_params(direction='in', length=4, width=1, colors='gray', which='minor')
z.minorticks_on()

s = "./Grids_%s.pdf" % (file_prefix)
sb.despine(offset=25, left=True, trim=True)
fig.tight_layout()
fig.savefig(s)
