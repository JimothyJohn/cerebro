import json
import pytest
import base64

def encode_image_to_base64(image_path: str) -> str:
    """
    Encodes the image to a Base64 string.

    Returns:
        str: Base64 encoded string of the image.
    """
    with open(image_path, 'rb') as image_file:
        encoded_string = base64.b64encode(image_file.read()).decode('utf-8')
    
    return encoded_string

@pytest.fixture()
def test_body() -> str:
    return json.dumps({
            "image": encode_image_to_base64("images/zidane.jpg"),
            "size": 640,
            "conf_thres": 0.7,
            "iou_thres": 0.5
            })
