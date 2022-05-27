#!/usr/bin/env bash

docker run -d --restart=unless-stopped \
    --name yolo-cpu \
    --shm-size=1g --ulimit memlock=-1 \
    --ulimit stack=67108864 \
    -p 5000:5000 \
    yolo-cpu:latest
