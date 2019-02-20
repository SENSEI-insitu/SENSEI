#!/bin/ksh

LOC=/work/camp/sensei

EXE="${LOC}/sensei/build/sensei/build-ascent/bin"

# Basic is the only data set using the BASIC_NAME type.
DATA_NAME="Basic"
 BASIC_NAME="uniform"      # 2d/3d
 BASIC_NAME="rectilinear"  # 2d/3d
 #BASIC_NAME="structured"   # 2d/3d
 #BASIC_NAME="tris"         # 2d
 #BASIC_NAME="quads"        # 2d
 #BASIC_NAME="polygons"     # Sensei does not support this cell type.
 #BASIC_NAME="tets"         # 3d
 #BASIC_NAME="hexs"         # 3d
 #BASIC_NAME="polyhedra"    # Sensei does not support this cell type.

 BASIC_DIMS="3,3,0"
 BASIC_DIMS="3,3,3"

# Other data sets.
#DATA_NAME="Braid"
#DATA_NAME="Spiral"

# The following data sets do not work
#DATA_NAME="Julia"
#DATA_NAME="Polytess"

CONFIG="/work/camp/sensei/sensei/build/sensei/test/conduitTest/ConduitTest${DATA_NAME}.xml"

export LD_LIBRARY_PATH="${LD_LIBRARY_PATH}:${LOC}/sensei/2.0.0-ascent/lib:${LOC}/vtk/8.1.1/lib:${LOC}/ascent/lib:${LOC}/ascent/build/conduit/install/lib:${LOC}/ascent/build/hdf5-1.8.16/install/lib:${LOC}/ascent/build/vtk-m/install/lib:${LOC}/ascent/build/vtk-h/install/lib:/work/tools/tbb/tbb-2019_U3/build/linux_intel64_gcc_cc5.3.1_libc2.22_kernel4.7.10_debug"

#GDB="gdb --args"
RUN_COM="${GDB} ${EXE}/conduitTest -i ${CONFIG} -d ${DATA_NAME} -b ${BASIC_NAME} -x ${BASIC_DIMS}"

echo $RUN_COM
$RUN_COM

