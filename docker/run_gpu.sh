#!/usr/bin/env bash

# cat /proc/driver/nvidia/version

docker run --gpus all -d --restart=unless-stopped \
    --name yolo-gpu \
    --shm-size=1g --ulimit memlock=-1 \
    --ulimit stack=67108864 \
    -p 5000:5000 \
    yolov5-cog:latest