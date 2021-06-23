#!/bin/bash

${SPACK_ROOT}/bin/spack load --sh $(${SPACK_ROOT}/bin/spack find --format "/{hash}") > spack_env_load.sh
export SAVE_PYTHON_PATH=${PYTHON_PATH}
export SAVE_LIBRARY_PATH=${LIBRARY_PATH}
export SAVE_LD_LIBRARY_PATH=${LD_LIBRARY_PATH}
export SAVE_PATH=${PATH}
export SAVE_CPATH=${CPATH}
export SAVE_MANPATH=${MANPATH}
source spack_env_load.sh
export PYTHON_PATH=${PYTHON_PATH}:${SAVE_PYTHON_PATH}
export LIBRARY_PATH=${LIBRARY_PATH}:${SAVE_LIBRARY_PATH}
export LD_LIBRARY_PATH=${LD_LIBRARY_PATH}:${SAVE_LD_LIBRARY_PATH}
export PATH=${PATH}:${SAVE_PATH}
export CPATH=${CPATH}:${SAVE_CPATH}
export MANPATH=${MANPATH}:${SAVE_MANPATH}

# using paraviews VTK
export VTK_DIR=${PARAVIEW_VTK_DIR}

export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$PWD:$PWD/lib
export PYTHONPATH=$PYTHONPATH:$PWD:$PWD/lib
export PATH=$PATH:$PWD:$PWD/bin

$@
