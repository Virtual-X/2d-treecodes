#! /bin/sh
#echo ARG1 is $1
#echo ARG2 is $2

IC_kernel()
{
OBJFILE=$1
KEYSTART=$2
KEYSTOP=$3

OPT=$(objdump -S $OBJFILE | egrep -n -i "(${KEYSTART}|${KEYSTOP})" | head -n 2 | tr "\n" " " | \
    awk 'BEGIN{ FS=":"} {printf( "%d,%dp", $1+1,$3-1);}')

RESULT=$(objdump -S $OBJFILE | sed -n ${OPT} | cut -d $'\t' -f3 | sed '/^$/d' | wc -l); 

printf "%d" "$RESULT"
}

DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"

RES=$(IC_kernel ${DIR}/*force-kernels.o "<force_e2p>" "<L_END_FORCE_E2P>")
printf " %b=%b" "-DE2P_IC" "$RES"

RES=$(IC_kernel ${DIR}/*force-kernels-tiled.o "<force_e2p_tiled>" "<L_END_FORCE_E2P_TILED>")
printf " %b=%b" "-DE2P_TILED_IC" "$RES" 

RES=$(IC_kernel ${DIR}/*downward-kernels.o "<L_DOWNWARD_E2L_ITERATION>" "<L_END_DOWNWARD_E2L_ITERATION>")
printf " %b=%b" "-DE2L_TILED_IC" "$RES" 
