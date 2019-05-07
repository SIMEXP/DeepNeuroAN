FROM tensorflow/tensorflow:1.13.0rc2-py3

LABEL maintainer="Loic Tetrel <loic.tetrel.pro@gmail.com>"

RUN useradd -ms /bin/bash jovyan

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

USER jovyan

COPY . /home/jovyan
