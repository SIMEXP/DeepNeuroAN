#!/bin/bash

echo "Will now build $1 with arg $2"

echo "Updating docker image from remote..."
git pull

echo "Building docker image..."
sudo docker build --tag=DeepNeuroAN

echo "Converting to singularity..."
sudo docker run -v /var/run/docker.sock:/var/run/docker.sock -v $(pwd):/output --privileged -t --rm singularityware/docker2singularity --name DeepNeuroAN DeepNeuroAN

echo "Transferring image to the server..."
rsync -rlt --info=progress2 DeepNeuroAN.simg stark.criugm.qc.ca:/data/cisl/CONTAINERS
