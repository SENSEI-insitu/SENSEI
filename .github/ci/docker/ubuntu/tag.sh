#!/bin/bash

source ${SPACK_ROOT}/share/spack/setup-env.sh

spack env activate ${SENSEI_ENV}
SPACK_LOCK=$(dirname $(spack config edit --print-file))/spack.lock
cat ${SPACK_LOCK} | python3 -c \
   'import json, sys; \
    obj = json.load(sys.stdin); \
    spec = list(obj["concrete_specs"].values())[0]; \
    print("-".join(spec["arch"].values()) + "-" + "-".join(spec["compiler"].values()));'
