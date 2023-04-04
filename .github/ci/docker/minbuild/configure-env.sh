. ${SPACK_ROOT}/share/spack/setup-env.sh

if [[ -z ${SENSEI_ENV} ]]; then
  echo "No environment configured."
  exit 1
fi

export CMAKE_CONFIGURATION=fedora35_ecp_catalyst
export CMAKE_CXX_COMPILER_LAUNCHER=ccache
export OMPI_ALLOW_RUN_AS_ROOT=1
export OMPI_ALLOW_RUN_AS_ROOT_CONFIRM=1

module use ${SPACK_ROOT}/share/spack/lmod/linux-fedora35-x86_64/Core

module load openmpi ninja swig
spack env activate ${SENSEI_ENV}

. ${SPACK_ROOT}/var/spack/environments/${SENSEI_ENV}/loads
