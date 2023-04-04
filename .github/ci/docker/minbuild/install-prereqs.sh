#!/bin/bash

set -xe

# Spack required packages
dnf install -y --setopt=install_weak_deps=False \
  python3 python3-pip \
  make texinfo libtool patch patchelf file tar git gnupg2 \
  gzip unzip bzip2 xz zstd \
  gcc g++ gfortran binutils ccache \
  Lmod which

python3 -m pip install clingo

# Install spack
if [[ -z ${SPACK_ROOT} ]]; then
  SPACK_ROOT=/opt/spack
fi
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
dnf install -y --setopt=install_weak_deps=False \
  bc

dnf clean all
