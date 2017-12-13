#!/bin/sh
T=100
N=64
k=8
beta=100
for i in {1..1}; do
	for Nq in "8"; do
		for p in "0.1"; do
			python -u biased_grid.py $T $N $beta $k $Nq $p &> "Biased_Grid Nq=$Nq p=$p $i" &
		done
	done
done
