ARG baseos=ubuntu-bionic
FROM spack/${baseos}:latest

# These are nice to have
RUN apt -yqq update && \
    apt -yqq install --no-install-recommends vim less bc && \
    rm -rf /var/lib/apt/lists/*

# Use spack@develop
RUN cd ${SPACK_ROOT} && \
    git init && \
    git remote add origin https://github.com/spack/spack.git && \
    git fetch origin develop && \
    git checkout -f develop && \
    rm -rf .git

# Setup the default configuration
COPY sensei-ci-packages.yaml $SPACK_ROOT/etc/spack/packages.yaml
RUN rm -rf /root/.spack && \
    spack compiler find --scope site

# Install dependencies
RUN spack install \
        -v -j$(grep -c '^processor' /proc/cpuinfo) --only dependencies \
        sensei && \
    spack clean -a

# Create the environment
RUN spack env create --without-view ci && \
    spack -e ci add $(spack find --format "/{hash}") && \
    spack -e ci install -v -j$(grep -c '^processor' /proc/cpuinfo) && \
    spack env activate ci  && \
    spack load $(spack find --format "/{hash}") && \
    spack env deactivate ci
