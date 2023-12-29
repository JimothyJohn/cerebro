# https://github.com/trainyolo/YOLO-ONNX/blob/main/yolo_onnx/utils.py
from PIL import Image
import numpy as np
import base64
import json


class ImageEncoder:
    def __init__(self, image_path: str):
        self.image_path = image_path

    def encode_to_base64(self) -> str:
        """
        Encodes the image to a Base64 string.

        Returns:
            str: Base64 encoded string of the image.
        """
        with open(self.image_path, "rb") as image_file:
            encoded_string = base64.b64encode(image_file.read())
            return encoded_string.decode("utf-8")

    def get_image_data_for_json(self) -> str:
        """
        Prepares the image data string for JSON object.

        Returns:
            str: Image data string formatted for JSON.
        """
        base64_data = self.encode_to_base64()
        return f"data:image/jpeg;base64,{base64_data}"


def main():
    parser = argparse.ArgumentParser(
        description="Encode an image to Base64 for HTTP requests."
    )
    parser.add_argument("image_path", type=str, help="Path to the image file")
    args = parser.parse_args()

    encoder = ImageEncoder(args.image_path)
    image_data_for_json = encoder.get_image_data_for_json()

    json_object = json.dumps({"image": image_data_for_json}, indent=4)
    print(json_object)


if __name__ == "__main__":
    main()


def prepare_input(img, size):
    # original size
    orig_w, orig_h = img.size

    # calc scale ratio
    ratio = min(size / orig_w, size / orig_h)

    # calc new size
    scaled_w, scaled_h = int(round(orig_w * ratio)), int(round(orig_h * ratio))

    # scale
    scaled_img = img.resize((scaled_w, scaled_h), resample=Image.Resampling.BILINEAR)

    # Calculate padding
    dh = 0 if (scaled_h % 32) == 0 else 32 - (scaled_h % 32)
    dw = 0 if (scaled_w % 32) == 0 else 32 - (scaled_w % 32)

    # Pad
    inp = np.full((scaled_h + dh, scaled_w + dw, 3), 114, dtype=np.float32)
    inp[:scaled_h, :scaled_w, :] = np.array(scaled_img)

    # Scale input pixel values to 0 to 1
    inp = inp / 255.0
    inp = inp.transpose(2, 0, 1)
    inp = inp[None, :, :, :]

    return inp, (orig_w, orig_h), (scaled_w, scaled_h)


def post_process(outp, conf_thres=0.7, iou_thres=0.5):
    preds = np.squeeze(outp[0]).T

    # Remove low-conf preds
    scores = np.max(preds[:, 4:], axis=1)
    keep = scores > conf_thres

    # get boxes, scores and class_ids
    preds = preds[keep, :]
    boxes = preds[:, :4]
    boxes = xywh2xyxy(boxes)

    scores = np.max(preds[:, 4:], axis=1)

    class_ids = np.argmax(preds[:, 4:], axis=1)

    # do multiclass nms
    indices = multiclass_nms(boxes, scores, class_ids, iou_thres=iou_thres)

    return boxes[indices], scores[indices], class_ids[indices]


def xywh2xyxy(boxes):
    new_boxes = np.copy(boxes)
    new_boxes[..., 0] = boxes[..., 0] - boxes[..., 2] / 2
    new_boxes[..., 1] = boxes[..., 1] - boxes[..., 3] / 2
    new_boxes[..., 2] = boxes[..., 0] + boxes[..., 2] / 2
    new_boxes[..., 3] = boxes[..., 1] + boxes[..., 3] / 2

    return new_boxes


def multiclass_nms(boxes, scores, class_ids, iou_thres=0.5):
    unique_ids = np.unique(class_ids)

    keep_boxes = []
    for class_id in unique_ids:
        class_indices = np.where(class_ids == class_id)[0]
        class_boxes = boxes[class_indices, :]
        class_scores = scores[class_indices]

        class_keep_boxes = nms(class_boxes, class_scores, iou_thres=iou_thres)
        keep_boxes.extend(class_indices[class_keep_boxes])

    return keep_boxes


def nms(boxes, scores, iou_thres=0.5):
    x1 = boxes[:, 0]
    y1 = boxes[:, 1]
    x2 = boxes[:, 2]
    y2 = boxes[:, 3]

    areas = (x2 - x1 + 1) * (y2 - y1 + 1)
    order = scores.argsort()[::-1]

    keep = []
    while order.size > 0:
        i = order[0]
        keep.append(i)

        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])

        w = np.maximum(0.0, xx2 - xx1 + 1)
        h = np.maximum(0.0, yy2 - yy1 + 1)
        inter = w * h
        ovr = inter / (areas[i] + areas[order[1:]] - inter)

        inds = np.where(ovr <= iou_thres)[0]
        order = order[inds + 1]

    return keep


def scale_boxes(boxes, orig_size, scaled_size):
    ox, oy, sx, sy = *orig_size, *scaled_size
    scale = np.array([ox / sx, oy / sy, ox / sx, oy / sy])
    boxes = boxes * scale
    return boxes


def parse_detections(boxes, scores, class_ids):
    detections = []
    for box, score, class_id in zip(boxes, scores, class_ids):
        detections.append(
            {
                "bbox": [int(b) for b in box],
                "score": float(round(score, 3)),
                "class_id": int(class_id),
            }
        )
    return detections
