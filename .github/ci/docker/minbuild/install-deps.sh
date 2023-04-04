#!/bin/bash

set -xe

source ${SPACK_ROOT}/share/spack/setup-env.sh

# make environment
spack env create ${SENSEI_ENV}
cp /sensei/tmp/spack.yaml ${SPACK_ROOT}/var/spack/environments/${SENSEI_ENV}
spack env activate ${SENSEI_ENV}

# buildcache
{
    HOST_IP_PORT=$(cat /sensei/tmp/buildcache-info.txt | awk '{ print $1 }') && \
    BUILDCACHE_PATH=$(cat /sensei/tmp/buildcache-info.txt | awk '{ print $2 }') && \
    curl -o /sensei/tmp/buildcache.zip http://${HOST_IP_PORT}/${BUILDCACHE_PATH} && \
    unzip /sensei/tmp/buildcache.zip -d /sensei/ && \
    spack mirror add sensei /sensei/buildcache && \
    spack buildcache update-index sensei
} || {
    echo "WARNING: No buildcache found, skipping."
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
