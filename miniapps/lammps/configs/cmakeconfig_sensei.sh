
export CC=`which gcc`
export CXX=`which g++`
export LAMMPS_DIR=`pwd`/src/lammps/



cmake -S ./src/SENSEI -B ./build/sensei \
  -DCMAKE_INSTALL_PREFIX=`pwd`/install/sensei \
  -DVTK_DIR=`pwd`/install/vtk/lib64/cmake/vtk-8.2 \
  -DENABLE_VTK_IO=ON \
  -DLAMMPS_DIR=$LAMMPS_DIR \
  -DENABLE_LAMMPS=ON \
  -DENABLE_MANDELBROT=OFF \
  -DENABLE_OSCILLATORS=OFF \
  -DENABLE_ADIOS2=ON \
  -DADIOS2_DIR=`pwd`/install/adios/lib64/cmake/adios2
