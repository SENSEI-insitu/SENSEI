#!/bin/bash

set -xe

export DEBIAN_FRONTEND="noninteractive"
apt-get update -y
apt-get install -y file git curl \
  build-essential software-properties-common \
  tar xz-utils bzip2 gzip unzip \
  python3 python3-pip ccache \
  patch patchelf libtool \
  vim

add-apt-repository ppa:ubuntu-toolchain-r/test -y
apt-get update -y
apt-get install -y gcc-11 g++-11 gfortran-11

# Install spack
mkdir -p ${SPACK_ROOT}
cd ${SPACK_ROOT}
git init
git remote add origin https://github.com/spack/spack.git
git fetch origin develop
git checkout -f develop

rm -rf .git

source ${SPACK_ROOT}/share/spack/setup-env.sh
spack compiler find --scope site

# Extra packages required by SENSEI test scripts (Not used by spack)
apt-get install -y bc
apt-get clean -y

# Setup sensei environment
spack env create ${SENSEI_ENV}
cp /sensei/tmp/spack.yaml ${SPACK_ROOT}/var/spack/environments/${SENSEI_ENV}
spack env activate ${SENSEI_ENV}
spack concretize -f
