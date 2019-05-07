#!/bin/bash

echo "Building docker image..."
sudo docker build --tag=deepneuroan .

echo "Converting to singularity..."
sudo docker run -v /var/run/docker.sock:/var/run/docker.sock -v $(pwd):/output --privileged -t --rm singularityware/docker2singularity --name deepneuroan deepneuroan

echo "Transferring image to the server..."
rsync -rlt --info=progress2 DeepNeuroAN.simg stark.criugm.qc.ca:/data/cisl/CONTAINERS
