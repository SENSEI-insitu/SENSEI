#!/bin/bash

set -e
set -x

source ${SPACK_ROOT}/share/spack/setup-env.sh
# Install Deps for a catayst based environment
spack install --fail-fast -v --only dependencies \
  -j$(grep -c '^processor' /proc/cpuinfo) \
  sensei+catalyst

spack clean -a

spack env create --without-view sensei
spack -e sensei add $(spack find --format "/{hash}")
spack -e sensei install

rm -rf /root/.spack

spack env activate sensei
spack env deactivate sensei
spack -e sensei env loads -m lmod
