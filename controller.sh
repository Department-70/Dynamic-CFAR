#!/bin/sh
# $1 file path to csv of params
# $2 kernal number being run
# $3 total kernels running
if ! [[ -f $1 ]];
then
	echo "Please input a real file"
	exit 1
fi
limit=$(cat $1 | wc -l)
count=0
while [ $count -lt $limit ] 
do
	cur=$(($count+$2))
	out=$(awk -F ',' -v cur=$cur 'NR==cur {print $1 " " $2 " " $3 " " $4 " " $4}' $1)
	count=$(($3+$count))
	./batchrun.sh $out
done




exit 1

