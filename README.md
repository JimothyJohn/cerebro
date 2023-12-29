![cerebro](docs/cerebro.png)

# Cerebro

Uses Replicate's 🐣 [Cog](https://github.com/replicate/cog) system to quickly deploy Ultralytic's 🚀 [PyTorch implementation](https://github.com/ultralytics/yolov5) of YOLOv5 as an AWS [Lambda function](https://aws.amazon.com/lambda/).

See [Cog](https://replicate.com/docs/creating-a-model#install-cog) and [YOLOv5](https://docs.ultralytics.com) for full documentation on training, testing and deployment.

## TODO 

[Rea tutorial](https://www.trainyolo.com/blog/deploy-yolov8-on-aws-lambda)

## Quickstart

<details open>
<summary>Convenience script for Ubuntu</summary>

```bash
# Install and run pretrained Cog
sudo utils/Install.sh
# Test endpoint
python3 utils/Endpoint.py
```

</details>

<details>
<summary>Prerequisites</summary>

[Install Docker](https://docs.docker.com/get-docker/) and then install the newest version of Cog

```bash
# Use Docker convenience script if you dare
curl -fsSL https://get.docker.com -o get-docker.sh
sudo sh get-docker.sh
# Download and configure Cog binary
sudo curl -o /usr/local/bin/cog -L https://github.com/replicate/cog/releases/latest/download/cog_`uname -s`_`uname -m`
sudo chmod +x /usr/local/bin/cog
```

</details>

<details>
<summary>(Optional) CPU Configuration</summary>

If you want to deploy with a CPU change line 2 in [cog.yml](cog.yml) to false:

```yml
build:
  gpu: false
```

</details>
<details open>
<summary>Inference</summary>

Test the [pretrained model](https://github.com/ultralytics/yolov5/releases) with

```bash
cog predict -i image=@data/images/zidane.jpg
```

This will output a JSON list of all detections by default:

```json
{
  "status": "succeeded",
  "output": [
    {
      "xmin":751.9083251953,
      "ymin":45.5722045898,
      "xmax":1148.5522460938,
      "ymax":716.2182006836,
      "confidence":0.8750465512,
      "class":0,
      "name":"person"
    },
    ...
  ]
}
```

You can also add an additional format argument to change the label output

```bash
cog predict -i image=@data/images/zidane.jpg -i format=yolo
```

</details>

<details>
<summary>Deployment</summary>

STARTER: Deploy the model locally at [http://localhost:5000](http://localhost:5000) using the below premade scripts.

```bash
# To run on GPU
docker/run_gpu.sh
# OR to run on CPU
docker/run_cpu.sh
# Send test image once the container is running
curl http://localhost:5000/predict -X POST -F input=@docs/zidane.jpg
```

You should see the same JSON output from the Inference step.

ADVANCED: Deploy in the cloud using the [Cloudflare Tunnel guide](DEPLOYMENT.md)

</details>

## To-do

- Add more label types
- Simplify JSON extraction
- Bridge to Jetson platform
- Training tutorial with Roboflow
- CI/CD to AWS Lambda
