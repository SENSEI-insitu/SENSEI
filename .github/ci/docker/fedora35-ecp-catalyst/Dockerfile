FROM fedora:35
MAINTAINER Ryan Krattiger <ryan.krattiger@kitware.com>

ENV SPACK_ROOT=/opt/spack

# Install Pre-reqs and Spack
COPY install-prereqs.sh /sensei/bin/
RUN /sensei/bin/install-prereqs.sh

ENV SPACK_PYTHON=/usr/bin/python3

# Install SENSEI deps with spack
COPY packages.yaml $SPACK_ROOT/etc/spack/
COPY modules.yaml $SPACK_ROOT/etc/spack/
COPY install-deps.sh /sensei/bin/
RUN /sensei/bin/install-deps.sh

WORKDIR /sensei/
COPY configure-env.sh /sensei/bin/
COPY launch-env.sh /sensei/bin/launcher
RUN echo ". /sensei/bin/configure-env.sh" >> ~/.bashrc

ENTRYPOINT []
CMD ["bash","--login"]
