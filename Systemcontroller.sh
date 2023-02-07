#!/bin/sh
# $1 file path to csv of tests to run
# $2 kernal number being run
# $3 total kernels running
if ! [[ -f $1 ]];
then
	echo "Please input a real file"
	exit 1
fi
IFS=', ' read -r -a array <<< "$(cat $1)"
count=0
limit="${#array[@]} "
while [ $count -lt $limit ] 
do
	cur=$(($count+$2))
	count=$(($3+$count))
	echo ${array[$cur]}
	#python bashrun.py array
	
done




exit 1

