#!/bin/bash

if [[ ! -f $(which docker) ]]; then
  echo "Cannot find docker."
  exit 1
fi

if [[ -z $1 ]]; then
  echo "Usage: build_image.sh <image-base-name>"
  exit 1
fi

if [ -f "$PWD/$1/Dockerfile" ]; then
  wdir=$PWD/$1
elif [ -f "$PWD/Dockerfile" ]; then
  wdir=$PWD
else
  echo "Cannot find Dockerfile: $1"
  exit 1
fi

# create image
tag_date=$(date +%Y%m%d)
tag=$1-$tag_date

base_container=ghcr.io/sensei-insitu/ci-ecp
buildcache_container=ghcr.io/sensei-insitu/sensei-buildcache

echo
echo "Building container: $base_container:$tag"
echo "  Dockerfile: $wdir/Dockerfile"

docker buildx create --name extended_log --use --driver-opt env.BUILDKIT_STEP_LOG_MAX_SIZE=100000000
docker buildx build --load -t $base_container:$tag-init-spack $wdir --target init_spack |& tee $1-init-spack-log.txt

BUILDCACHE_VERSION=$(docker run --rm -it $base_container:$tag-init-spack /sensei/bin/tag.sh | tr -d '\r')

IMAGE_EXISTS=$(docker manifest inspect $buildcache_container:$BUILDCACHE_VERSION | grep "layers")
if [[ -z ${IMAGE_EXISTS} ]]; then
  echo "Buildcache version $BUILDCACHE_VERSION not found, using empty buildcache."
  BUILDCACHE_VERSION="empty"
fi

echo "Buildcache version: $BUILDCACHE_VERSION"
docker buildx build --load -t $base_container:$tag $wdir --build-arg BUILDCACHE_VERSION=$BUILDCACHE_VERSION --target build_sensei |& tee $1-build-sensei-log.txt
docker push $base_container:$tag

docker rmi $base_container:$tag-init-spack
