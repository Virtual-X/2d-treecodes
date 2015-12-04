#! /bin/sh
#echo ARG1 is $1
#echo ARG2 is $2

OPT=$(objdump -S $2 |egrep -n -i "(END_$1|<$1>)" | tr "\n" " " | awk 'BEGIN{ FS=":"} {printf( "%d,%dp", $1+1,$3-1);}')

RESULT=$(objdump -S $2 | sed -n ${OPT} | cut -d $'\t' -f3 | sed '/^$/d' | wc -l); 

printf "%d" "$RESULT"