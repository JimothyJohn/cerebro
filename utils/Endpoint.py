#!/usr/bin/env python
import requests


def process_image(mediaURL):
    """Extract media from the message and send to AI endpoint"""
    resp_str = requests.post(
        url="http://localhost:5000/predictions",
        json={"input": {"image": mediaURL}},
    ).json()
    print(f"Yolo response: {resp_str}")

    return resp_str["status"]

process_image("https://thefoodxp.com/wp-content/uploads/2019/12/1920_tulsaopening2of4-837991.jpg")
