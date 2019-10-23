#!/bin/ksh

LOC=/work/camp/sensei

EXE=${LOC}/sensei/3.0.0-ascent/bin

SENSEI_INSTALL=${LOC}/sensei/3.0.0-ascent
VTK_INSTALL=${LOC}/vtk/8.1.1
ASCENT_INSTALL=${LOC}/ascent
CONDUIT_INSTALL=${LOC}/ascent/build/conduit/install
HDF5_INSTALL=${LOC}/ascent/build/hdf5-1.8.16/install
VTKM_INSTALL=${LOC}/ascent/build/vtk-m/install
VTKH_INSTALL=${LOC}/ascent/build/vtk-h/install
TBB_INSTALL=/work/tools/tbb/tbb-2019_U3/build/linux_intel64_gcc_cc5.3.1_libc2.22_kernel4.7.10_debug

export LD_LIBRARY_PATH="${LD_LIBRARY_PATH}:${SENSEI_INSTALL}/lib:${VTK_INSTALL}/lib:${ASCENT_INSTALL}/lib:${CONDUIT_INSTALL}/lib:${HDF5_INSTALL}/lib:${VTKM_INSTALL}/lib:${VTKH_INSTALL}/lib:${TBB_INSTALL}"

#GDB="gdb --args"
#MPI_EXE="mpirun -n 1"
RUN_COM="${GDB} ${MPI_EXE} ${EXE}/ascent_render_example"

echo $RUN_COM
$RUN_COM

