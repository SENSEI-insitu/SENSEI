#!/bin/bash

set -xe

source ${SPACK_ROOT}/share/spack/setup-env.sh

# make environment
spack env create ${SENSEI_ENV}
cp /sensei/tmp/spack.yaml ${SPACK_ROOT}/var/spack/environments/${SENSEI_ENV}
spack env activate ${SENSEI_ENV}

# download and install buildcache
{
    curl -o /sensei/tmp/buildcache.tar.gz https://data.kitware.com/api/v1/item/643867e22f922a9d90bb2a05/download && \
    tar -xzf /sensei/tmp/buildcache.tar.gz -C /sensei/ && \
    mv /sensei/buildcache-ubuntu /sensei/buildcache/ && \
    spack mirror add sensei /sensei/buildcache && \
    spack buildcache update-index sensei
} || {
    echo "Error: Failed to install buildcache."
    exit 1
}

# install
N_THREADS=$(grep -c '^processor' /proc/cpuinfo)
spack concretize -f
spack install -v --use-cache --no-check-signature -j ${N_THREADS} --only dependencies
spack install -v --use-cache --no-check-signature -j ${N_THREADS} sensei

# cleanup
spack clean -a
rm -rf /root/.spack /sensei/tmp /sensei/buildcache

# load modules
spack module lmod refresh -y
spack env loads -m lmod

spack env deactivate
