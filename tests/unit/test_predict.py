import pytest
from unittest.mock import patch, MagicMock
from cerebro.predict import Predictor, AutoModelForCausalLM, AutoProcessor, torch
import requests
from contextlib import contextmanager
from cog import Path

TEST_IMAGE_PATH = Path(
    "https://3.bp.blogspot.com/_bVtGlUaW-tA/TIYjKLOSIfI/AAAAAAAAOFg/NjhGhwLBWcE/s1600/3.jpg"
)


@pytest.fixture
def predictor():
    return Predictor()


@pytest.fixture
def mock_setup(predictor):
    with patch.object(predictor, "setup"):
        yield


@pytest.fixture
def mock_model(predictor):
    with patch.object(predictor, "model", create=True) as mock:
        mock.generate.return_value = torch.tensor([[1, 2, 3]])
        yield mock


@pytest.fixture
def mock_processor(predictor):
    with patch.object(predictor, "processor", create=True) as mock:
        mock.batch_decode.return_value = ["Mocked response"]
        yield mock


@contextmanager
def mock_image_retrieval(status_code=200):
    with patch("cerebro.predict.requests.get") as mock_get, patch(
        "cerebro.predict.Image.open"
    ) as mock_image_open:
        mock_response = MagicMock()
        mock_response.status_code = status_code
        mock_response.raw = MagicMock()
        mock_get.return_value = mock_response

        mock_image = MagicMock()
        mock_image_open.return_value.convert.return_value = mock_image
        yield mock_get, mock_image_open


@pytest.mark.parametrize(
    "image, prompt, max_new_tokens, temperature, do_sample, expected_error",
    [
        (Path(TEST_IMAGE_PATH), "Describe this image", 1024, 0.7, True, None),
        (Path(TEST_IMAGE_PATH), "What's in this image?", 2048, 0.5, True, None),
        # (Path(""), "Empty image path", 1024, 0.7, True, ValueError),
        (Path(" "), "Empty image path", 1024, 0.7, True, ValueError),  # Added this case
        (Path(TEST_IMAGE_PATH), "", 1024, 0.7, True, ValueError),
        (Path(TEST_IMAGE_PATH), "Test prompt", 63, 0.7, True, ValueError),
        (Path(TEST_IMAGE_PATH), "Test prompt", 2049, 0.7, True, ValueError),
        (Path(TEST_IMAGE_PATH), "Test prompt", 1024, -0.1, True, ValueError),
        (Path(TEST_IMAGE_PATH), "Test prompt", 1024, 1.1, True, ValueError),
    ],
)
def test_predict_schema(
    mock_setup,
    mock_model,
    mock_processor,
    predictor,
    image,
    prompt,
    max_new_tokens,
    temperature,
    do_sample,
    expected_error,
):
    with mock_image_retrieval():
        if expected_error:
            with pytest.raises(expected_error):
                predictor.predict(
                    image=image,
                    prompt=prompt,
                    max_new_tokens=max_new_tokens,
                    temperature=temperature,
                    do_sample=do_sample,
                )
        else:
            result = predictor.predict(
                image=image,
                prompt=prompt,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                do_sample=do_sample,
            )
            assert isinstance(result, str)
            assert result == "Mocked response"

        if not expected_error:
            predictor.processor.assert_called_once()
            predictor.model.generate.assert_called_once()


@pytest.mark.parametrize(
    "image_urls, prompt, expected_response",
    [
        (TEST_IMAGE_PATH, "Describe this image", "Mocked response"),
        (
            "http://example.com/image1.jpg,http://example.com/image2.jpg",
            "Describe these images",
            "Mocked response for multiple images",
        ),
    ],
)
def test_predict_images(
    mock_setup,
    mock_model,
    mock_processor,
    predictor,
    image_urls,
    prompt,
    expected_response,
):
    with mock_image_retrieval():
        mock_processor.batch_decode.return_value = [expected_response]
        result = predictor.predict(image_urls, prompt)
        assert result == expected_response


def test_predict_model_generation(mock_setup, mock_model, mock_processor, predictor):
    with mock_image_retrieval():
        result = predictor.predict(
            TEST_IMAGE_PATH,
            "Test prompt",
            max_new_tokens=100,
            temperature=0.5,
            do_sample=False,
        )

        predictor.model.generate.assert_called_once()
        call_args = predictor.model.generate.call_args[1]
        assert call_args["max_new_tokens"] == 100
        assert call_args["temperature"] == 0.5
        assert call_args["do_sample"] == False
        assert result == "Mocked response"


def test_predict_empty_response(mock_setup, predictor):
    with mock_image_retrieval(), patch.object(
        predictor, "model", create=True
    ) as mock_model, patch.object(
        predictor, "processor", create=True
    ) as mock_processor:

        mock_model.generate.return_value = torch.tensor([[]])
        mock_processor.batch_decode.return_value = [""]

        result = predictor.predict(TEST_IMAGE_PATH, "Describe this image")
        assert result == ""


@patch("cerebro.predict.AutoModelForCausalLM.from_pretrained")
@patch("cerebro.predict.AutoProcessor.from_pretrained")
@patch("cerebro.predict.torch.cuda.is_available")
@patch("cerebro.predict.is_ampere_or_newer")
def test_setup(
    mock_is_ampere, mock_cuda_available, mock_processor, mock_model, predictor
):
    mock_cuda_available.return_value = True
    mock_is_ampere.return_value = True
    mock_model.return_value = MagicMock(spec=AutoModelForCausalLM)
    mock_processor.return_value = MagicMock(spec=AutoProcessor)

    predictor.setup()

    mock_model.assert_called_once_with(
        "/model-cache",
        device_map="auto",
        torch_dtype=torch.float16,
        trust_remote_code=True,
        _attn_implementation="flash_attention_2",
    )
    mock_processor.assert_called_once_with(
        "/model-cache", trust_remote_code=True, num_crops=4
    )
    assert predictor.model is not None
    assert predictor.processor is not None
    assert torch.backends.cudnn.benchmark
    assert torch.backends.cudnn.enabled
