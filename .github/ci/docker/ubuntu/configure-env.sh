#!/bin/bash
. ${SPACK_ROOT}/share/spack/setup-env.sh

SPACK_LMOD_ROOT=$(spack find --format {prefix} lmod)
. ${SPACK_LMOD_ROOT}/lmod/lmod/init/bash

if [[ -z ${SENSEI_ENV} ]]; then
  echo "No environment configured."
  exit 1
fi

export CMAKE_CONFIGURATION=ubuntu_ecp_catalyst
export CMAKE_CXX_COMPILER_LAUNCHER=ccache
export OMPI_ALLOW_RUN_AS_ROOT=1
export OMPI_ALLOW_RUN_AS_ROOT_CONFIRM=1

spack env activate ${SENSEI_ENV}

module use ${SPACK_ROOT}/share/spack/lmod/linux-ubuntu20.04-x86_64/Core
module load openmpi ninja swig

. /sensei/loads
