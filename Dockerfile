ARG TAG

FROM tensorflow/tensorflow:2.2.0rc3${TAG}

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

# joblib space error
# https://github.com/datmo/datmo/issues/237
ENV JOBLIB_TEMP_FOLDER=/tmp

COPY . /DeepNeuroAN
