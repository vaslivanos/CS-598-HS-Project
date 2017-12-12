#!/bin/sh
T=50
N=36
k=15
beta=100
for i in {1..1}; do
	for Nq in "10"; do
		for p in "0.05"; do
			python -u unbiased_grid.py $T $N $beta $k $Nq $p &> "Unbiased_Grid Nq=$Nq p=$p $i" &
		done
	done
done
