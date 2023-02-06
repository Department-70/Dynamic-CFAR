#!/bin/sh
# $1 should be alg param
# $2 should be PFA param
# $3 should be dist param
# $4 should be A list param
mkdir ./results/

PTH="./results/$1_$2_$3_$4.txt"
cat batch.txt > $PTH
echo "--label=$1_$2_$3_$4" >> $PTH
echo "--alg=$1" >> $PTH
echo "--dist=$3" >> $PTH
echo "--target_PFA=$2" >> $PTH
echo "--a_list" >> $PTH
arr=($4)
for i in ${arr[@]}
do
	echo $i >> $PTH
done
echo "--TH_init" >> $PTH
for i in ${arr[@]}; do echo 0.5 >> $PTH; done
echo "--delta" >> $PTH
for i in ${arr[@]}; do echo 0.1 >> $PTH; done
#python bashrun.py $PTH $
