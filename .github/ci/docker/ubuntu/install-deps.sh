#!/bin/bash

set -xe

source ${SPACK_ROOT}/share/spack/setup-env.sh
spack env activate ${SENSEI_ENV}
{
    ls /sensei/buildcache/build_cache && \
    spack mirror add sensei /sensei/buildcache && \
    spack buildcache update-index sensei
} || {
    echo "Error: Failed to install buildcache."
    echo "  Skipping buildcache installation..."
}

# install
N_THREADS=$(grep -c '^processor' /proc/cpuinfo)
spack install -v --use-cache --no-check-signature -j ${N_THREADS}
spack install -v --use-cache --no-check-signature -j ${N_THREADS} sensei

# cleanup
spack clean -a
rm -rf /root/.spack /sensei/tmp /sensei/buildcache

# load modules
spack module lmod refresh -y
spack env loads -m lmod
