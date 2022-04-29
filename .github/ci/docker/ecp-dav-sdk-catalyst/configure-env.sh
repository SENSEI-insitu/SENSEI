. ${SPACK_ROOT}/share/spack/setup-env.sh

if [[ ! -z ${SENSEI_ENV} ]]; then
  module use ${SPACK_ROOT}/share/spack/lmod/linux-fedora35-x86_64/Core
  module load openmpi
  . ${SPACK_ROOT}/var/spack/environments/${SENSEI_ENV}/loads
else
  echo "No environment configured."
fi
