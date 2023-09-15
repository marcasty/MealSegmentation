# FROM python:3.8
FROM nvidia/cuda:12.2.0-devel-ubuntu22.04
LABEL maintainer="Pranav Iyer pranaviyer2@gmail.com"
LABEL version="1.0"

RUN APT_INSTALL="apt-get install -y --no-install-recommends --alow-unauthenticated" \
    && apt update \
    && DEBIAN_FRONTEND=noninteractive $APT_INSTALL \
    git \
    build-essential \
    libboost-all-dev \
    python3-colcon-common-extensions \
    python3-dev \
    python3-tk \
    cmake \
    libgli-mesa-glx \
    libsm6 \
    libxext6 \
    libglib \
    && rm -rf /var/lib/apt/lists/* \
    && apt-get clean

ADD requirements.txt /

RUN pip install -r /requirements.txt

WORKDIR /me
ENV PYTHONUNBUFFERED=1
ENV CUDA_HOME=/usr/local/cuda/lib64