
export LD_LIBRARY_PATH=/lus/grand/projects/visualization/srizzi/BUILDS/sensei-lammps/install/vtk/lib64:/lus/grand/projects/visualization/srizzi/BUILDS/sensei-lammps/install/adios/lib64:$LD_LIBRARY_PATH

#mpiexec -n 4 ./lammpsDriver in.rhodo -n 5 -sensei config.xml
#mpiexec -n 4 ./lammpsDriver in.rhodo -n 5 -sensei adioswrite.xml
./SENSEIEndPoint -a vtkosc.xml -t adiostransport.xml
