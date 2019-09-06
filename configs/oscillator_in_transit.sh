#!/bin/bash

n=4
b=64
dt=10
delay=10
max_delay=10
file=random_2d_${b}.bp
bld=`echo -e '\e[1m'`
red=`echo -e '\e[31m'`
grn=`echo -e '\e[32m'`
blu=`echo -e '\e[36m'`
wht=`echo -e '\e[0m'`


#echo "+ module load sensei/2.1.1-catalyst-shared"
#module load sensei/2.1.1-catalyst-shared

module load paraview/5.5.2
module load adios/1.13.1

set -x

cat configs/adios_write.xml | sed "s/.*/$blu&$wht/"

mpiexec -n ${n} \
    ./oscillator -b ${n} -t ${dt} -s ${b},${b},1  -p 0 -g 1 \
    -f configs/adios_write.xml \
    configs/random_2d_${b}.osc 2>&1 | sed "s/.*/$red&$wht/" &

cat configs/python.xml | sed "s/.*/$blu&$wht/"

mpiexec -n ${n} \
    ./ADIOS1EndPoint -r flexpath \
    -f configs/python.xml \
    random_2d_${b}.bp  2>&1 | sed "s/.*/$grn&$wht/"
