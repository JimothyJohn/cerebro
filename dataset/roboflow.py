import requests
import urllib.parse
import os

ROBOFLOW_API_KEY = os.environ.get("ROBOFLOW_API_KEY")

def upload_annotation(annotation, id):
    upload_url = f"https://api.roboflow.com/dataset/whataburger/annotate/{id}"

    params = {
        "api_key": ROBOFLOW_API_KEY,
        "name": f"{id}.txt",
    }

    headers = {
        "Content-Type": "text/plain",
    }

    r = requests.post(
        upload_url, params=params, data=annotation, headers=headers
    ).json()

    return r


def upload_image(uri, filename, results):

    # Construct the URL
    upload_url = "https://api.roboflow.com/dataset/whataburger/upload"

    params = {
        "api_key": ROBOFLOW_API_KEY,
        "name": f"{filename}.jpg",
        "split": "train",
        "image": urllib.parse.quote_plus(uri),
    }

    # POST to the API
    r = requests.post(upload_url, params=params).json()

    return r["id"]
