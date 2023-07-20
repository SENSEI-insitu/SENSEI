#!/bin/bash

export CC=`which gcc`
export CXX=`which g++`


cmake -S ./src/ADIOS2 -B ./build/adios \
 -DCMAKE_INSTALL_PREFIX=`pwd`/install/adios \
 -DADIOS2_USE_Fortran=OFF \
 -DADIOS2_BUILD_EXAMPLES=OFF \
 -DCMAKE_BUILD_TYPE=Debug
