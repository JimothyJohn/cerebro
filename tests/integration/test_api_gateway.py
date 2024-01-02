import os
import boto3
import pytest
import requests
import json
from tests.utils import *
from time import time

class TestApiGateway:

    @pytest.fixture()
    def api_gateway_url(self):
        """ Get the API Gateway URL from Cloudformation Stack outputs """
        stack_name = os.environ.get("AWS_SAM_STACK_NAME")

        if stack_name is None:
            raise ValueError('Please set the AWS_SAM_STACK_NAME environment variable to the name of your stack')

        client = boto3.client("cloudformation")

        try:
            response = client.describe_stacks(StackName=stack_name)
        except Exception as e:
            raise Exception(
                f"Cannot find stack {stack_name} \n" f'Please make sure a stack with the name "{stack_name}" exists'
            ) from e

        stacks = response["Stacks"]
        stack_outputs = stacks[0]["Outputs"]
        api_outputs = [output for output in stack_outputs if output["OutputKey"] == "CerebroApi"]

        if not api_outputs:
            raise KeyError(f"HelloWorldAPI not found in stack {stack_name}")

        return api_outputs[0]["OutputValue"]  # Extract url from stack outputs

    def test_get_error(self, api_gateway_url):
        """ Call the API Gateway endpoint and check the response """
        response = requests.get(api_gateway_url)

        assert response.status_code == 400
        assert response.json() == {"message": "Error: This endpoint only accepts POST requests."}

    def test_post_error(self, api_gateway_url):
        """ Call the API Gateway endpoint and check the response """
        response = requests.post(api_gateway_url, data="")

        assert response.status_code == 500
        assert response.json() == {"message": 'the JSON object must be str, bytes or bytearray, not NoneType'}

    def test_image_error(self, api_gateway_url):
        """ Call the API Gateway endpoint and check the response """
        response = requests.post(api_gateway_url, data=json.dumps({
            "image": "images/zidane.jpg",
            "size": 640,
            "conf_thres": 0.7,
            "iou_thres": 0.5
            }))

        assert response.status_code == 500
        assert response.json()["message"].startswith("cannot identify image file") 

    def test_good_image(self, api_gateway_url, test_body):
        """ Call the API Gateway endpoint and check the response """
        start_time = time()
        response = requests.post(api_gateway_url, data=test_body)
        response_time = time() - start_time
        cost = response_time * 1000 * 0.0000000167
        body = response.json()

        # Basic Assertions
        assert response.status_code == 200
        assert "detections" in body
        assert isinstance(body["detections"], list)
        assert len(body["detections"]) > 0

        # Check each detection
        for detection in body["detections"]:
            assert "bbox" in detection
            assert "score" in detection
            assert "class_name" in detection
            assert isinstance(detection["bbox"], list)
            assert isinstance(detection["score"], float)
            assert isinstance(detection["class_name"], str)
            assert len(detection["bbox"]) == 4  # Typically, bbox should have 4 values

        # Performance Assertion
        max_allowed_time = 1.0  # seconds, adjust as needed
        assert response_time < max_allowed_time, f"Response time exceeded {max_allowed_time} seconds"
