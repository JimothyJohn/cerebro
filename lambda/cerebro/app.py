# https://github.com/trainyolo/YOLOv8-aws-lambda/blob/main/lambda-codebase/app.py
import os
import json
import base64
from io import BytesIO
from PIL import Image
from yolo_onnx.yolov8_onnx import YOLOv8

yolov8_detector = YOLOv8(
    os.path.join(os.path.dirname(os.path.realpath(__file__)), "models", "yolov8n.onnx")
)


def detect(body):
    # get params
    img_b64 = body["image"]
    SIZE = 640
    conf_thres = body.get("conf_thres", 0.7)
    iou_thres = body.get("iou_thres", 0.5)

    # open image
    img = Image.open(BytesIO(base64.b64decode(img_b64.encode("ascii"))))
    img_resized = img.resize((SIZE, SIZE))  # Resize image to the expected size


    # infer result
    detections = yolov8_detector(
        img_resized, size=SIZE, conf_thres=conf_thres, iou_thres=iou_thres
    )

    return detections


def lambda_handler(event, context):
    """Sample pure Lambda function

    Parameters
    ----------
    event: dict, required
        API Gateway Lambda Proxy Input Format

        Event doc: https://docs.aws.amazon.com/apigateway/latest/developerguide/set-up-lambda-proxy-integrations.html#api-gateway-simple-proxy-for-lambda-input-format

    context: object, required
        Lambda Context runtime methods and attributes

        Context doc: https://docs.aws.amazon.com/lambda/latest/dg/python-context-object.html

    Returns
    ------
    API Gateway Lambda Proxy Output Format: dict

        Return doc: https://docs.aws.amazon.com/apigateway/latest/developerguide/set-up-lambda-proxy-integrations.html
    """

    # try:
    #     ip = requests.get("http://checkip.amazonaws.com/")
    # except requests.RequestException as e:
    #     # Send some context about this error to Lambda Logs
    #     print(e)

    #     raise e

    # Check if the incoming request is a POST request
    if event['httpMethod'] != 'POST':
        return {
            "statusCode": 400,
            "body": json.dumps({"message": "Error: This endpoint only accepts POST requests."})
        }

    try:
        # Assuming detect is a function you've defined to process the incoming request
        detections = detect(json.loads(event["body"]))
        return {
            "statusCode": 200,
            "body": json.dumps({"detections": detections})
        }
    except Exception as e:
        # Handle other exceptions here as needed
        print(e)
        return {
            "statusCode": 500,
            "body": json.dumps({"message": str(e)})
        }
