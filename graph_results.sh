#!/bin/sh
# Change Nq and/or P to 'all' to remove filter
Nq="1"
P="0.01"

python graph_files.py $Nq $P
