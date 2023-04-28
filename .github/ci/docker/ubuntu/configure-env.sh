#!/bin/bash
. ${SPACK_ROOT}/share/spack/setup-env.sh

# source the bootstraped lmod
. /bootstrap/runner/install/linux-ubuntu20.04-x86_64/gcc-11.1.0/lmod-8.7.2-2dgwl2hnw3ztoij4yibuhimturloucbz/lmod/8.7.2/init/bash

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

. ${SPACK_ROOT}/var/spack/environments/${SENSEI_ENV}/loads
