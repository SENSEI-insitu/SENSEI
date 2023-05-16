ARG BUILDCACHE_VERSION=empty
FROM ubuntu:20.04 AS init_spack

ENV SPACK_PYTHON=/usr/bin/python3
ENV SPACK_ROOT=/opt/spack
ENV SENSEI_ENV=sensei

# Install Pre-reqs and Spack
COPY install-prereqs.sh /sensei/tmp/
COPY spack.yaml /sensei/tmp/
COPY modules.yaml $SPACK_ROOT/etc/spack/
RUN /sensei/tmp/install-prereqs.sh
COPY tag.sh /sensei/bin/

# TODO: buildcache logic
FROM ghcr.io/willdunklin/sensei-buildcache:$BUILDCACHE_VERSION AS buildcache

# Install SENSEI deps with spack
FROM init_spack AS build_sensei
# TODO: need to test this for edge cases. does this work if there is no buildcace?
COPY --from=buildcache /buildcache /sensei/buildcache
COPY install-deps.sh /sensei/tmp/
RUN /sensei/tmp/install-deps.sh

WORKDIR /sensei/
COPY configure-env.sh /sensei/bin/
COPY launch-env.sh /sensei/bin/
RUN echo ". /sensei/bin/configure-env.sh" >> ~/.bashrc
