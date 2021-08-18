# buster is required to use perf
#FROM python:3.9-buster

#FROM ubuntu:focal

# required for GPUs
FROM tensorflow/tensorflow:latest-gpu-jupyter

ARG USERNAME=python
# On Linux, replace with your actual UID, GID if not the default 1000
ARG USER_UID=1000
ARG USER_GID=$USER_UID

# buster (required for perf)
#RUN apt update && apt install sudo time docker.io linux-perf -y && rm -rf /var/lib/apt/lists/*

# nvidia/cuda:11.0-base-ubuntu20.04 (required for GPU)
# follow the instructions here for setting up nvidia container toolkit on HOST
# https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html
#RUN apt update && apt install sudo time docker.io gpustat python3 python3-pip -y && rm -rf /var/lib/apt/lists/*
ENV DEBIAN_FRONTEND=noninteractive 
RUN apt update && apt-get install -y --no-install-recommends tzdata && rm -rf /var/lib/apt/lists/*
RUN apt update && apt install python3-opencv -y && rm -rf /var/lib/apt/lists/*

# Create the user
RUN groupadd --gid $USER_GID $USERNAME \
    && useradd --uid $USER_UID --gid $USER_GID -m $USERNAME && echo $USERNAME":"$USERNAME | chpasswd && adduser $USERNAME sudo \
    && mkdir -p /home/$USERNAME/.vscode-server /home/$USERNAME/.vscode-server-insiders \
    && chown ${USER_UID}:${USER_GID} /home/$USERNAME/.vscode-server*

# ********************************************************
# * Anything else you want to do like clean up goes here *
# ********************************************************

RUN python3 -m pip install --upgrade pip
RUN apt-get update && apt-get install -y protobuf-compiler wget && rm -rf /var/lib/apt/lists/*

# change the username
USER $USERNAME
ENV PATH="${PATH}:/home/${USERNAME}/.local/bin"

# if using root user in the container
ENV PATH="${PATH}:/root/.local/bin"
