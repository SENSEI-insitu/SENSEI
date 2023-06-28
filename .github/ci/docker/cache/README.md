# SENSEI Spack Buildcache

This directory contains the spack binary buildcache for SENSEI's dependencies.

Use of buildcaches speeds up the build process for iterating SENSEI CI containers.

## Extracting a buildcache from a working CI image

SENSEI uses CI docker images to perform automated testing. These CI images are built
using the `build_image.sh` script. Occasionally development needs to be performed on
the CI image themselves, which can be an arduous process given the need to rebuild
SENSEI and all of its dependencies in Spack. To speed up this process, SENSEI uses
buildcaches to speed up the build process.

The `extract_buildcache.sh` script will extract the buildcache from a built SENSEI
CI docker image and create a new minimal docker image containing the buildcache.

To use the `extract_buildcache.sh` script, specify the source CI image you'd like to
extract built binaries from. If no image is specified, the script will use the most
recent `sensei-insitu/ci-ecp` image found locally. The script will then extract the
buildcache to a local volume, and run a docker built to create a new buildcache image.
This image is tagged with the operating system, compiler and architecture of the
source CI image to ensure compatibility.

## How buildcaches are used in SENSEI CI images

SENSEI CI images are built with the `build_image.sh` script. This script builds a CI
docker image based on a preexisting buildcache docker image. The script will identify
the proper buildcache image tag for the CI environment, and attempt to use the
prebuilt binaries if they are available. If the buildcache image is not in the
container registry, it defaults to using the `empty` tagged image.

`build_image.sh` invokes a docker build that uses the buildcache image as an
intermediate layer to mount copy the buildcache into the CI image. The buildcache is
then used speed up the to build the SENSEI CI image. The buildcache is not included
in the final image.
