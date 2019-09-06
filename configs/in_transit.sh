#!/bin/bash

n=4
b=64
dt=10
delay=10
max_delay=10
file=random_2d_${b}.bp
iterations=2
xcells=16
ycells=16
xblocks=2
yblocks=2
x0=0
x1=10
y0=0
y1=10
t0=1
t1=10
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

cat configs/write_adios1_flexpath.xml | sed "s/.*/$blu&$wht/"

mpiexec -n ${n} \
    python testPartitionersWrite.py configs/write_adios1_flexpath.xml ${iterations} \
     ${xblocks} ${yblocks} ${xcells} ${ycells} ${x0} ${x1} ${y0} ${y1} ${t0} ${t1} \
     2>&1 | sed "s/.*/$red&$wht/" &

cat configs/python.xml | sed "s/.*/$blu&$wht/"

mpiexec -n ${n} \
    ./ADIOS1EndPoint -r flexpath \
    -f configs/python.xml \
    random.bp  2>&1 | sed "s/.*/$grn&$wht/"
