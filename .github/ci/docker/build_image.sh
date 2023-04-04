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

# use buildcache if specified
if [[ -n $2 && -f "./$2" ]]; then
  echo "Using buildcahce './$2'"

  # setup python server for remote buildcache
  BRIDGE_EXISTS=$(docker network ls | grep -c ' *bridge *bridge')
  if [[ $BRIDGE_EXISTS -eq 0 ]]; then
    docker network ls
    echo "Network 'bridge' does not exist. Please create it for the python server to work."
    exit 1
  fi

  HOST_IP=$(docker network inspect bridge | python3 -c "import sys, json; print(json.load(sys.stdin)[0]['IPAM']['Config'][0]['Gateway'])")
  HOST_PORT=12345
  echo "$HOST_IP:$HOST_PORT $2" > $wdir/buildcache-info.txt

  python3 -m http.server $HOST_PORT -b $HOST_IP &
  pid=$!
else
  echo "Warning: No buildcache found '$2', skipping..."
  echo "  To use a buildcache, run 'build_image.sh <image-base-name> < dbuildcache>',"
  echo "  where <buildcache> is a zip file containing a spack buildcache"
fi

# create image
base_name=$1
tag_date=$(date +%Y%m%d)
tag=$1-$tag_date

echo
echo "Building container: senseiinsitu/ci:$tag"
echo "  Dockerfile: $wdir/Dockerfile"
docker build -t senseiinsitu/ci:$tag $wdir |& tee $1-build-log.txt

if [[ -n $pid ]]; then
  kill $pid
  echo "Killed python server with pid $pid."
fi

# docker push senseiinsitu/ci:$tag
