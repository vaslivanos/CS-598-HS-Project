#!/bin/sh
# Change Nq and/or P to 'all' to remove filter
Nq="10"
P="0.1"

python graph_files.py "Unbiased" $Nq $P
