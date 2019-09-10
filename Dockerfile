ARG TAG

FROM tensorflow/tensorflow:2.0.0rc0${TAG}-py3

LABEL maintainer="Loic Tetrel <loic.tetrel.pro@gmail.com>"

RUN apt-get update && apt-get install -y \
    wget \
    git \
    htop \
    graphviz

RUN pip3 install SimpleITK \
    pybids \	
    sklearn \
    scipy \
    nilearn \
    numpy \
    pyquaternion

RUN mkdir /DeepNeuroAN
RUN mkdir /DATA

COPY . /DeepNeuroAN
