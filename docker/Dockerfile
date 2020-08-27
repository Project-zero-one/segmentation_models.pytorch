FROM pytorch/pytorch:1.4-cuda10.1-cudnn7-devel

RUN apt-get update
RUN apt-get install -y software-properties-common locales vim git
# For OpenCV
RUN apt-get install -y libsm6 libxrender1 libxext-dev libgtk2.0-dev
# fish shell
RUN apt-add-repository ppa:fish-shell/release-3 -y
RUN apt-get update
RUN apt-get install -y fish

RUN locale-gen en_US.UTF-8
ENV LANG en_US.UTF-8
ENV LANGUAGE en_US:en
ENV LC_ALL en_US.UTF-8

RUN pip install --upgrade pip
RUN pip install \
    opencv-python \
    pandas \
    scikit-learn \
    matplotlib \
    albumentations \
    natsort

RUN pip install resnest --no-deps

WORKDIR /
