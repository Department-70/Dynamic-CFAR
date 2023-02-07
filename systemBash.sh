#!/bin/sh
# $1 file path to csv of test to run
# $2 number of kernals to run
if ! [[ -f $1 ]];
then
	echo "Please input a real file"
	exit 1
fi
if [ $2 -ge 7 ]
then
	echo "More then 7 kernals not recomended, if you really want to do this, go into the code to change it" 
	exit 1
fi
count=0
while [ $count -lt $2 ] 
do
	echo $count
	./Systemcontroller.sh $1 $count $2 &
	count=$((1+$count))
done




exit 1

