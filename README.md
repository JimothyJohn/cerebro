![cerebro](docs/cerebro.png)

# Cerebro

[![Replicate](https://replicate.com/jimothyjohn/phi3-vision-instruct/badge)](https://replicate.com/jimothyjohn/phi3-vision-instruct)

Cerebro is a perception endpoint that analyzes visual data using continuously better, more generalized AI models. 

### Engine

The current engine is Microsoft's multi-modal [Phi-3.5-vision-instruct](https://huggingface.co/microsoft/Phi-3.5-vision-instruct) hosted on a [Replicate](https://replicate.com/) NVIDIA [A100](https://www.nvidia.com/en-us/data-center/a100/) endpoint. Phi3.5 seemed to offer the best balance of performance and speed as of September 2024. Previous endpoint was the more precise, but narrow [YOLOv8](https://docs.ultralytics.com/models/yolov8/).

## Getting Started

To run locally you should use an NVIDIA GPU with at least 8GB of VRAM.

```bash
# Clone the repository
git clone https://github.com/JimothyJohn/cerebro.git
cd cerebro 
# Use convenience script (will take a few minutes to build the first time)
bash Quickstart
```

### Notes

* **Most of the code in this repository is AI-generated and should be reviewed for correctness and compatibility.** You can find the prompt that generated most of this code in [PROMPT.md](PROMPT.md)

* Call Cerebro from an edge device with the [Ojito](https://github.com/JimothyJohn/ojito) repository.

* Build your own self-hosted project on AWS with the Cerebro [reverse proxy](https://github.com/JimothyJohn/cerebro-reverse-proxy) repository.

### Contributing

Contributions are welcome! Please submit a pull request or open an issue for any changes or suggestions.

### License - MIT
