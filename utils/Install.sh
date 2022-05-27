#!/usr/bin/env bash

# Install Docker via convenience script
curl -fsSL https://get.docker.com -o get-docker.sh
sudo sh get-docker.sh

# Install latest version of Cog
sudo curl -o /usr/local/bin/cog -L https://github.com/replicate/cog/releases/latest/download/cog_`uname -s`_`uname -m`
sudo chmod +x /usr/local/bin/cog

# Build and run Yolov5
sudo cog build -t yolo-cpu
sudo docker/run_cpu.sh
