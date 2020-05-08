#!/bin/bash

echo "Removing previous container"
yes | rm *.simg

echo "Building docker image..."
if sudo docker build --build-arg TAG=$1 --tag=deepneuroan$1 .; then
	echo "Converting to singularity..."
	sudo docker run -v /var/run/docker.sock:/var/run/docker.sock -v $(pwd):/output --privileged -t --rm singularityware/docker2singularity --name deepneuroan$1 deepneuroan$1

	echo "Deleting none images"
	docker rmi --force $(docker images | grep none | awk '{ print $3; }')

	echo "Transferring image to elm..."
	rsync -rlt --info=progress2 deepneuroan$1.simg elm.criugm.qc.ca:/data/cisl/CONTAINERS
	# echo "Transferring image to cedar..."
	# rsync -rlt --info=progress2 deepneuroan$1.simg cedar.computecanada.ca:~/projects/rrg-pbellec/CONTAINERS
else
    echo "Docker build was not successfull"
fi
