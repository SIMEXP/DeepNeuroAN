#!/bin/bash

echo "Removing previous container"
rm deepneuroan$1

echo "Building docker image..."
sudo docker build --build-arg TAG=$1 --tag=deepneuroan$1 .

echo "Converting to singularity..."
sudo docker run -v /var/run/docker.sock:/var/run/docker.sock -v $(pwd):/output --privileged -t --rm singularityware/docker2singularity --name deepneuroan$1 deepneuroan$1

echo "Deleting none images"
docker rmi --force $(docker images | grep none | awk '{ print $3; }')

echo "Transferring image to the server..."
rsync -rlt --info=progress2 deepneuroan$1.simg stark.criugm.qc.ca:/data/cisl/CONTAINERS
