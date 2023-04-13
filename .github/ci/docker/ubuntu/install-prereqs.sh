#!/bin/bash

set -xe

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

apt update -y
apt install -y ccache
# Extra packages required by SENSEI test scripts (Not used by spack)
apt install -y bc
apt clean -y
