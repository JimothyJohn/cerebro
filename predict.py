import cog
import torch
from io import BytesIO
import json


def yolo_labels(results):
    ogHeight, ogWidth, _ = results.imgs[0].shape
    detections = json.loads(results.pandas().xyxy[0].to_json(orient="records"))
    labels = ""
    for detection in detections:
        classification = detection["class"]
        width = detection["xmax"] - detection["xmin"]
        height = detection["ymax"] - detection["ymin"]
        x = detection["xmin"] + width / 2
        y = detection["ymin"] + height / 2
        labels = f"{labels}{classification} {x/ogWidth} {y/ogHeight} {width/ogWidth} {height/ogHeight}\n"

    return labels


class Predictor(cog.BasePredictor):
    def setup(self):
        """Load the model into memory to make running multiple predictions efficient"""
        self.net = torch.hub.load("ultralytics/yolov5", "yolov5s")
        # Alternatively use a custom model
        # self.net = torch.hub.load("ultralytics/yolov5", "custom", "./<your-weights-here>.pt")

    # Define the input types for a prediction
    def predict(
        self,
        image: cog.Path = cog.Input(description="RGB input image"),
        format: str = cog.Input(description="Format of label", default="json"),
    ) -> str:
        """Run a single prediction on the model"""
        # ... pre-processing ...
        results = self.net(image)
        output = results.pandas().xyxy[0].to_json(orient="records")
        if format == "yolo":
            output = yolo_labels(results)
        return output
