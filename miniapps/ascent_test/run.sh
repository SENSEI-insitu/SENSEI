#!/bin/ksh

LOC=/work/camp/sensei

EXE=${LOC}/sensei/build/sensei/build-ascent/bin

export LD_LIBRARY_PATH="${LD_LIBRARY_PATH}:${LOC}/sensei/2.0.0-ascent/lib:${LOC}/vtk/8.1.1/lib:${LOC}/ascent/lib:${LOC}/ascent/build/conduit/install/lib:${LOC}/ascent/build/hdf5-1.8.16/install/lib:${LOC}/ascent/build/vtk-m/install/lib:${LOC}/ascent/build/vtk-h/install/lib:/work/tools/tbb/tbb-2019_U3/build/linux_intel64_gcc_cc5.3.1_libc2.22_kernel4.7.10_debug"

#GDB="gdb --args"
RUN_COM="${GDB} ${EXE}/ascent_render_example"

echo $RUN_COM
$RUN_COM

