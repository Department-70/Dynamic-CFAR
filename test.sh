#!/bin/sh
# $1 should be alg param
# $2 should be PFA param
# $3 should be dist param
# $4 should be A list min param
# $5 should be A list max param
mkdir ./results/

PTH="./results/$1_$2_$3_$4_$5.txt"
cat batch.txt > $PTH
echo "--label=$1_$2_$3_$4_$5" >> $PTH
echo "--alg=$1" >> $PTH
echo "--dist=$3" >> $PTH
echo "--target_PFA=$2" >> $PTH
echo "--a_list" >> $PTH
echo $4 >> $PTH
echo $5 >> $PTH

python bashrun.py $PTH &
exit 1

