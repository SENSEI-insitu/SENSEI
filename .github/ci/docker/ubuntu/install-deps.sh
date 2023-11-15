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

# cleanup
spack clean -a
rm -rf /root/.spack /sensei/tmp /sensei/buildcache

# generate modules
spack module lmod refresh -y

SENSEI_MODULE=$(spack module lmod find --full-path sensei)
# write sensei module dependencies to file
cat ${SENSEI_MODULE} | grep depends_on | awk -F'"' '{ print "module load " $2 }' > /sensei/loads

# remove temporary sensei build from image
spack uninstall -y sensei
