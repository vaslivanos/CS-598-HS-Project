#!/usr/bin/python
#/* -.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.
#* Purpose : Grant Proposal on Resource-Constrained Networks Simulation
#* Creation Date : 14-11-2017
#* Last Modified : Tue 14 Nov 2017 12:38:31 CST
#* Created By : Vasilis Livanos <basilis3761@yahoo.gr>
#_._._._._._._._._._._._._._._._._._._._._.*/

import seaborn as sb
import matplotlib.pyplot as plt
import numpy as np
import itertools
import sys

# Turn interactive mode off
plt.ioff()

with open(sys.argv[1]) as f:
    content = f.readlines()
    content = [x.strip() for x in content] 

#print(content)

s = content[0]
T = int((content[1].split())[2])
N = int((content[1].split())[5])
k = int((content[1].split())[11])
Nq = int((content[1].split())[14])
s += (", N: %s, k: %s, Q: %s" % (N, k, Nq))

opt = float((content[2].split())[2])
#print(opt)

p1 = float((content[4].split())[2])
#print(p1)

p2 = float((content[4+T+2].split())[2])
#print(p2)

q1 = [float((content[5+i].split())[1]) for i in range(T)]
#print(q1)

q2 = [float((content[7+T+i].split())[1]) for i in range(T)]
#print(q2)

if(int(sys.argv[2]) == 1):
    fig, ax = sb.mpl.pyplot.subplots(1, 1)
    colors = sb.mpl_palette('magma', n_colors=2)

    ax.plot(
            range(T),
            q1,
            label=(r'$p={0}$'.format(p1)),
            color=colors[0],
            linestyle='solid',
            linewidth=1)
    ax.plot(
            range(T),
            q2,
            label=(r'$p={0}$'.format(p2)),
            color=colors[1],
            linestyle='dotted',
            linewidth=1)
    z = ax
    z.set_ylabel(r"Quality / OPT")
    z.set_xlabel(r"Time")
    z.legend(fontsize=6)
    z.set_xlim([0, T])
    z.set_xticks(np.arange(0, T, 1.0 * T / 10), minor=True)
    z.set_ylim([0, 1])
    z.set_yticks(np.arange(0, 1, 0.05), minor=True)
    z.tick_params(direction='in', length=6, width=2, colors='k', which='major', labelsize=8)
    z.tick_params(direction='in', length=4, width=1, colors='gray', which='minor')
    z.minorticks_on()

    fig.suptitle(s)
    sb.despine(offset=25, left=True, trim=True)
    fig.tight_layout()
    s += ".pdf"
    fig.savefig(s)
else:
    dq1 = [0]
    dq1 += [j-i for i, j in zip(q1[:-1], q1[1:])]
    dq1 = list(itertools.accumulate(dq1))
    dq2 = [0]
    dq2 += [j-i for i, j in zip(q2[:-1], q2[1:])]
    dq2 = list(itertools.accumulate(dq2))
    #print(opt - opt*q1[0])
    #print(opt - opt*q2[0])
    dq1 = [1.0 * x / (1 - q1[0]) for x in dq1]
    dq2 = [1.0 * x / (1 - q2[0]) for x in dq2]
    #print(dq1)
    #print(dq2)

    fig, ax = sb.mpl.pyplot.subplots(1, 1)
    colors = sb.mpl_palette('magma', n_colors=2)

    ax.plot(
            range(T),
            dq1,
            label=(r'$p={0}$'.format(p1)),
            color=colors[0],
            linestyle='solid',
            linewidth=1)
    ax.plot(
            range(T),
            dq2,
            label=(r'$p={0}$'.format(p2)),
            color=colors[1],
            linestyle='dotted',
            linewidth=1)
    z = ax
    z.set_ylabel(r"Derivative of Quality")
    z.set_xlabel(r"Time")
    z.legend(fontsize=6)
    z.set_xlim([0, T])
    z.set_xticks(np.arange(0, T, 1.0 * T / 10), minor=True)
    z.set_ylim([0, 1])
    z.set_yticks(np.arange(0, 1, 0.05), minor=True)
    z.tick_params(direction='in', length=6, width=2, colors='k', which='major', labelsize=8)
    z.tick_params(direction='in', length=4, width=1, colors='gray', which='minor')
    z.minorticks_on()

    fig.suptitle(s)
    sb.despine(offset=25, left=True, trim=True)
    fig.tight_layout()
    s += " ds.pdf"
    fig.savefig(s)
