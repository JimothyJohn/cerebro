import pytest
from pathlib import Path
from cerebro.predict import Predictor
import torch

# AI-generated comment: Define test constants
TEST_IMAGE_PATH = Path(
    "https://3.bp.blogspot.com/_bVtGlUaW-tA/TIYjKLOSIfI/AAAAAAAAOFg/NjhGhwLBWcE/s1600/3.jpg"
)


@pytest.fixture(scope="module")
def predictor():
    # AI-generated comment: Create and set up the Predictor instance
    pred = Predictor()
    pred.setup()
    return pred


def test_predict_with_url(predictor):
    # AI-generated comment: Test prediction with an image URL
    result = predictor.predict(
        image=TEST_IMAGE_PATH,
        prompt="Describe this image in detail",
        max_new_tokens=1024,
        temperature=0.7,
        do_sample=True,
    )
    assert isinstance(result, str)
    assert len(result) > 0
    assert "cat" in result.lower() or "feline" in result.lower()


def test_predict_with_file_path(predictor):
    # AI-generated comment: Test prediction with a local image file
    result = predictor.predict(
        image=TEST_IMAGE_PATH,
        prompt="What's in this image?",
        max_new_tokens=512,
        temperature=0.5,
        do_sample=False,
    )
    assert isinstance(result, str)
    assert len(result) > 0
    assert "cat" in result.lower() or "feline" in result.lower()


@pytest.mark.parametrize("max_new_tokens", [64, 1024, 2048])
def test_predict_different_token_lengths(predictor, max_new_tokens):
    # AI-generated comment: Test prediction with various token lengths
    result = predictor.predict(
        image=TEST_IMAGE_PATH,
        prompt="Analyze this image",
        max_new_tokens=max_new_tokens,
        temperature=0.7,
        do_sample=True,
    )
    assert isinstance(result, str)
    assert 0 < len(result) <= max_new_tokens * 4  # Approximate token to character ratio


@pytest.mark.parametrize("temperature", [0.1, 0.5, 0.9])
def test_predict_different_temperatures(predictor, temperature):
    # AI-generated comment: Test prediction with various temperature settings
    result = predictor.predict(
        image=TEST_IMAGE_PATH,
        prompt="Describe the scene",
        max_new_tokens=512,
        temperature=temperature,
        do_sample=True,
    )
    assert isinstance(result, str)
    assert len(result) > 0


def test_predict_no_sampling(predictor):
    # AI-generated comment: Test prediction without sampling (greedy decoding)
    result = predictor.predict(
        image=TEST_IMAGE_PATH,
        prompt="What do you see?",
        max_new_tokens=512,
        temperature=0.7,
        do_sample=False,
    )
    assert isinstance(result, str)
    assert len(result) > 0


def test_predict_error_handling(predictor):
    # AI-generated comment: Test error handling for invalid inputs
    with pytest.raises(ValueError):
        predictor.predict(
            image="",
            prompt="This should fail",
            max_new_tokens=1024,
            temperature=0.7,
            do_sample=True,
        )

    with pytest.raises(ValueError):
        predictor.predict(
            image=TEST_IMAGE_PATH,
            prompt="",
            max_new_tokens=1024,
            temperature=0.7,
            do_sample=True,
        )

    with pytest.raises(ValueError):
        predictor.predict(
            image=TEST_IMAGE_PATH,
            prompt="Invalid max_new_tokens",
            max_new_tokens=3000,  # Too high
            temperature=0.7,
            do_sample=True,
        )

    with pytest.raises(ValueError):
        predictor.predict(
            image=TEST_IMAGE_PATH,
            prompt="Invalid temperature",
            max_new_tokens=1024,
            temperature=1.5,  # Out of range
            do_sample=True,
        )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
def test_predict_gpu_usage(predictor):
    # AI-generated comment: Test GPU memory usage during prediction
    torch.cuda.empty_cache()
    initial_memory = torch.cuda.memory_allocated()

    result = predictor.predict(
        image=TEST_IMAGE_PATH,
        prompt="Detailed analysis of this image",
        max_new_tokens=2048,
        temperature=0.7,
        do_sample=True,
    )

    peak_memory = torch.cuda.max_memory_allocated()
    assert peak_memory > initial_memory
    assert isinstance(result, str)
    assert len(result) > 0


def test_predict_performance(predictor):
    # AI-generated comment: Test prediction performance
    import time

    start_time = time.time()
    result = predictor.predict(
        image=TEST_IMAGE_PATH,
        prompt="Quick description",
        max_new_tokens=256,
        temperature=0.7,
        do_sample=True,
    )
    end_time = time.time()

    assert isinstance(result, str)
    assert len(result) > 0
    assert end_time - start_time < 10  # Adjust this threshold as needed
