# predict.py
# Prediction interface for Cog ⚙️
# https://github.com/replicate/cog/blob/main/docs/python.md
"""
This program is for Phi-3.5-vision-instruct, the multi-modal generation of Phi models.
AI-generated code, please review for correctness.
"""

from cog import BasePredictor, Input, Path
import time
import torch
from PIL import Image
from transformers import AutoModelForCausalLM, AutoProcessor
import requests
import logging
from typing import List, Dict, Any
import urllib.parse

# Configure logging
logging.basicConfig(level=logging.WARNING)
logger = logging.getLogger(__name__)

# MODEL_NAME = "microsoft/Phi-3.5-vision-instruct"
MODEL_CACHE = "/model-cache"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def is_ampere_or_newer() -> bool:
    if torch.cuda.is_available():
        capability = torch.cuda.get_device_capability()
        return capability[0] >= 8  # Ampere is compute capability 8.0+
    return False


class Predictor(BasePredictor):
    model: AutoModelForCausalLM
    processor: AutoProcessor

    def setup(self) -> None:
        """Load the model into memory to make running multiple predictions efficient"""
        if is_ampere_or_newer():
            torch.backends.cudnn.benchmark = True  # Optimize CuDNN kernel selection
            torch.backends.cudnn.enabled = True  # Ensure CuDNN is enabled

        start_time = time.time()
        logger.info("Setting up the model...")
        try:
            # Load the model from the cache directory
            self.model = AutoModelForCausalLM.from_pretrained(
                MODEL_CACHE,
                device_map="auto",
                torch_dtype=torch.float16,
                trust_remote_code=True,
                _attn_implementation=(
                    "flash_attention_2" if is_ampere_or_newer() else "eager"
                ),  # or 'eager' if flash_attn not installed
            )

            # self.model = torch.jit.script(self.model)  # For dynamic control flow

            # For best performance, use num_crops=4 for multi-frame, num_crops=16 for single-frame.
            self.processor = AutoProcessor.from_pretrained(
                MODEL_CACHE,
                trust_remote_code=True,
                # Experiment with lower numbers for faster processing
                num_crops=4,  # multi-frame
                # num_crops=16  # single-frame, results in torch.OutOfMemoryError: CUDA out of memory
            )
            logger.info(f"Model loaded in {time.time() - start_time:.2f} seconds")
        except Exception as e:
            logger.error(f"Error during model setup: {e}")
            raise e

    @torch.inference_mode()
    def predict(
        self,
        image: Path = Input(description="Image path, URL, or file path"),
        prompt: str = Input(description="Input prompt"),
        max_new_tokens: int = Input(
            description="Max new tokens", default=1024, ge=64, le=2048
        ),
        temperature: float = Input(
            description="Temperature for generation", default=0.7, ge=0.0, le=1.0
        ),
        do_sample: bool = Input(
            # Set to False if enough VRAM is available
            description="Whether or not to use sampling; use greedy decoding otherwise.",
            default=True,
        ),
    ) -> str:
        # Extract values from FieldInfo objects if necessary
        image = getattr(image, "default", image)
        prompt = getattr(prompt, "default", prompt)
        max_new_tokens = getattr(max_new_tokens, "default", max_new_tokens)
        temperature = getattr(temperature, "default", temperature)
        do_sample = getattr(do_sample, "default", do_sample)

        # Input validation
        if not image or str(image).strip() == "":
            raise ValueError("Image path, URL, or file path cannot be empty")
        if not prompt or str(prompt).strip() == "":
            raise ValueError("Prompt cannot be empty")
        if max_new_tokens < 64 or max_new_tokens > 2048:
            raise ValueError("max_new_tokens must be between 64 and 2048")
        if temperature < 0 or temperature > 1:
            raise ValueError("Temperature must be between 0 and 1")

        try:
            # Process image
            if urllib.parse.urlparse(str(image)).scheme in ("http", "https"):
                # It's a URL
                response: requests.Response = requests.get(str(image), stream=True)
                response.raise_for_status()
                img = Image.open(response.raw).convert("RGB")
            else:
                # It's a local file or path
                img = Image.open(image).convert("RGB")

            # Prepare messages
            messages: List[Dict[str, str]] = [
                {"role": "user", "content": "<|image_1|>\n" + prompt},
            ]

            # Prepare prompt
            prepared_prompt: str = self.processor.tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )

            # Prepare inputs
            inputs: Dict[str, Any] = self.processor(
                prepared_prompt, images=[img], return_tensors="pt"
            ).to(DEVICE)

            # Generation arguments
            generation_args: Dict[str, Any] = {
                "max_new_tokens": max_new_tokens,
                "temperature": temperature,
                "do_sample": do_sample,
                "eos_token_id": self.processor.tokenizer.eos_token_id,
            }

            # Generate output
            outputs: torch.Tensor = self.model.generate(**inputs, **generation_args)

            # Remove input tokens
            generated_tokens: torch.Tensor = outputs[:, inputs["input_ids"].shape[1] :]

            # Decode response
            response: str = self.processor.batch_decode(
                generated_tokens,
                skip_special_tokens=True,
                clean_up_tokenization_spaces=False,
            )[0]

            return response

        except Exception as e:
            logger.error(f"Error during prediction: {e}")
            raise
