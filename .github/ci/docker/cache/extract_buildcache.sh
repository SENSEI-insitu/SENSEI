#!/bin/bash

cd $(dirname $0)

IMAGE=$1
if [[ -z $1 ]]; then
    echo "No image specified. Using latest sensei-insitu/ci-ecp image."
    IMAGE=$(docker image list | grep sensei-insitu/ci-ecp | awk '{ print $1 ":" $2 }' | head -n 1)
fi
echo "Using image: ${IMAGE}"
echo

tag=$(docker run -it --rm --volume "$(pwd):/sensei/cache" ${IMAGE} /sensei/bin/tag.sh | tr -d '\r')
buildcache_container=ghcr.io/willdunklin/sensei-buildcache

IMAGE_EXISTS=$(docker manifest inspect $buildcache_container:$tag | grep "layers")
if [[ -n ${IMAGE_EXISTS} ]]; then
    echo "Found existing buildcache version: $tag"
    echo "Appending to image."

    # clear the buildcache folder
    docker run -it --rm --volume "$(pwd):/cache" alpine rm -rf /cache/buildcache

    container=$(docker run --detach $buildcache_container:$tag)
    docker cp $container:/buildcache ./buildcache
    docker rm -f $container
else
    echo "Buildcache version $tag not found, using empty buildcache."
fi

# generate the buildcache
docker run -it --rm --volume "$(pwd):/sensei/cache" ${IMAGE} \
    bin/launch-env.sh \
    spack buildcache create --force --allow-root --unsigned /sensei/cache/buildcache

docker build -t $buildcache_container:$tag .
docker push $buildcache_container:$tag

# cleanup the buildcache directory, sometimes this has root permissions
docker run -it --rm --volume "$(pwd):/cache" alpine rm -rf /cache/buildcache
