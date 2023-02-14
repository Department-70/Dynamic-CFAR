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
	if [[ -f "kill.txt" ]]; 
	then
		echo "Killing runs ${array[$cur]} onword. If this was a mistake, please remove kill.txt, and rerun"
		exit 1
	fi
	echo "Controller running on ${array[$cur]}"
	python System_Parser_SC.py @sys.txt --exp_index "${array[$cur]}"
	
done




exit 1

