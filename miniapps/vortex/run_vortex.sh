#!/bin/sh
rm *.bp *.vti

# If dims are 256,32,32:

# This happens to make 2508 patches
#mpirun -np 8 ./vortex -f vortex.xml -i 11 -r 4 -l 4 -log

# This happens to make 1722 patches
#mpirun -np 8 ./vortex -f vortex.xml -i 11 -r 2 -l 4 -log

# This happens to make 11 patches
mpirun -np 2 ./vortex -f vortex.xml -i 11 -r 2 -l 2 
