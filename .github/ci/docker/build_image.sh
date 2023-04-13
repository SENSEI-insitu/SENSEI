#!/bin/bash

if [[ ! -f $(which docker) ]]; then
  echo "Cannot find docker."
  exit 1
fi

if [[ -z $1 ]]; then
  echo "Usage: build_image.sh <image-base-name> [buildcache]"
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
base_name=$1
tag_date=$(date +%Y%m%d)
tag=$1-$tag_date

echo
echo "Building container: senseiinsitu/ci:$tag"
echo "  Dockerfile: $wdir/Dockerfile"
docker build -t ghcr.io/willdunklin/sensei:$tag $wdir |& tee $1-build-log.txt
# docker push ghcr.io/willdunklin/sensei:$tag

# docker build --no-cache -t senseiinsitu/ci:$tag $wdir |& tee $1-build-log.txt
