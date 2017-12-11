#!/bin/sh
# Change Nq and/or P to 'all' to remove filter
Nq="all"
P="all"

python graph_files.py "Biased" $Nq $P
