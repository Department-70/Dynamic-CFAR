#!/bin/sh
# $1 file path to csv of params
# $2 number of kernals to run
if ! [[ -f $1 ]];
then
	echo "Please input a real file"
	exit 1
fi
if [ $2 -ge 5 ]
then
	echo "More then 4 kernals not recomended, if you really want to do this, go into the code to change it" 
	exit 1
fi
limit=$(cat $1 | wc -l)
count=1
while [ $count -le $2 ] 
do
	echo $count
	./controller.sh $1 $count $2 &
	count=$((1+$count))
done




exit 1

