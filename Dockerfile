FROM tensorflow/tensorflow:1.13.0rc2-py3

LABEL maintainer="Loic Tetrel <loic.tetrel.pro@gmail.com>"

COPY . .

RUN apt-get update && apt-get install -y \
    wget \
    git \
    htop

RUN pip3 install SimpleITK \
    pybids \	
    sklearn \
    scipy \
    nilearn \
    numpy
