# https://github.com/trainyolo/YOLOv8-aws-lambda/blob/main/lambda-codebase/app.py
import os
import json
import base64
from io import BytesIO
from typing import Dict, List, Any
from PIL import Image
import yaml
from yolov8_onnx import YOLOv8


class CocoClassMapper:
    def __init__(self, coco_yaml_path: str):
        self.class_id_to_name = self._load_class_names(coco_yaml_path)

    @staticmethod
    def _load_class_names(yaml_path: str) -> Dict[int, str]:
        """
        Loads the class ID to name mapping from the coco.yaml file.

        :param yaml_path: Path to the coco.yaml file.
        :return: A dictionary mapping class IDs to their names.
        """
        with open(yaml_path, "r") as file:
            data = yaml.safe_load(file)
            return {int(k): v for k, v in data["names"].items()}

    def map_class_names(self, detections: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Maps class IDs in the detections to their corresponding class names.

        :param detections: A list of detection dictionaries.
        :return: The enriched list of detections with class names.
        """
        for detection in detections:
            class_id = detection.get("class_id")
            detection["class_name"] = self.class_id_to_name.get(class_id, "Unknown")
        return detections


yolov8_detector = YOLOv8(
    os.path.join(os.path.dirname(os.path.realpath(__file__)), "models", "yolov8n.onnx")
)

mapper = CocoClassMapper(
    os.path.join(os.path.dirname(os.path.realpath(__file__)), "coco.yaml")
)


def detect(body: Dict[str, Any], class_mapper: CocoClassMapper) -> Dict[str, Any]:
    """
    Perform object detection on an image and return detections with class names.

    :param body: A dictionary containing the image and detection parameters.
    :param class_mapper: An instance of CocoClassMapper to map class IDs to names.
    :return: Detections with class names.
    """
    # Get parameters
    img_b64 = body["image"]
    SIZE = 640
    conf_thres = body.get("conf_thres", 0.7)
    iou_thres = body.get("iou_thres", 0.5)

    # Open and resize image
    img = Image.open(BytesIO(base64.b64decode(img_b64.encode("ascii"))))
    img_resized = img.resize((SIZE, SIZE))

    # Infer result
    detections = yolov8_detector(
        img_resized, size=SIZE, conf_thres=conf_thres, iou_thres=iou_thres
    )

    # Map class IDs to names
    detections_with_names = class_mapper.map_class_names(detections)

    return detections_with_names


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
    if event["httpMethod"] != "POST":
        return {
            "statusCode": 400,
            "body": json.dumps(
                {"message": "Error: This endpoint only accepts POST requests."}
            ),
        }

    try:
        # Assuming detect is a function you've defined to process the incoming request
        detections = detect(json.loads(event["body"]), mapper)
        return {"statusCode": 200, "body": json.dumps({"detections": detections})}
    except Exception as e:
        # Handle other exceptions here as needed
        print(e)
        return {"statusCode": 500, "body": json.dumps({"message": str(e)})}
