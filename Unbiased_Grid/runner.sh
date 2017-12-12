#!/bin/sh
T=100
N=64
k=50
beta=100
for i in {1..1}; do
	for Nq in "1" "2" "10"; do
		for p in "0.01" "0.05" "0.1"; do
			python -u unbiased_grid.py $T $N $beta $k $Nq $p &> "Unbiased_Grid Nq=$Nq p=$p $i" &
		done
	done
done
