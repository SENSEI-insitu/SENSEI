FROM ecpe4s/ubuntu20.04-runner-x86_64:2023-03-01

ENV SPACK_PYTHON=/usr/bin/python3
ENV SPACK_ROOT=/opt/spack
ENV SENSEI_ENV=sensei

# Install Pre-reqs and Spack
COPY install-prereqs.sh /sensei/tmp/
RUN /sensei/tmp/install-prereqs.sh

# Install SENSEI deps with spack
COPY spack.yaml /sensei/tmp/
COPY modules.yaml $SPACK_ROOT/etc/spack/
COPY install-deps.sh /sensei/tmp/
RUN /sensei/tmp/install-deps.sh

WORKDIR /sensei/
COPY configure-env.sh /sensei/bin/
COPY launch-env.sh /sensei/bin/
RUN echo ". /sensei/bin/configure-env.sh" >> ~/.bashrc