#!/bin/bash

cd /tmp
git clone https://github.com/IDEA-Research/GroundingDINO.git
cd GroundingDINO
git checkout -q 57535c5a79791cb76e36fdb64975271354f10251
CUDA_HOME=/usr/local/cuda-11.7 pip install -e .
