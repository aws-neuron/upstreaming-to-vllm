import base64
import binascii
import functools
import logging
import signal
from io import BytesIO
import torch
import numpy as np
from torchvision import transforms
from PIL import Image, UnidentifiedImageError
from PIL.Image import DecompressionBombError, DecompressionBombWarning, Image as PILImage
from fastapi.exceptions import RequestValidationError

IMAGE_METADATA_TOO_LARGE_ERROR = 'Decompressed Data Too Large'
IMAGE_TOKEN_DENOMINATOR = 750

logger = logging.getLogger(__name__)


def timeout(seconds=10, error_cls=TimeoutError, error_message="Timed out"):
    def decorator(func):
        def _handle_timeout(signum, frame):
            raise error_cls(error_message)

        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            signal.signal(signal.SIGALRM, _handle_timeout)
            signal.alarm(seconds)
            try:
                result = func(*args, **kwargs)
            finally:
                signal.alarm(0)
            return result
        return wrapper

    return decorator


@timeout(seconds=5, error_cls=RequestValidationError, error_message="IMAGE_TOO_LARGE")
def decompress_image(image_b64: str, try_load_image=False):
    try:
        decoded_bytes = base64.b64decode(image_b64, validate=True)
    except binascii.Error:
        raise RequestValidationError("INVALID_BASE_64_STR")
    except ValueError:
        raise RequestValidationError("INVALID_BASE_64_STR")
    image_data = BytesIO(decoded_bytes)
    try:
        image = Image.open(image_data)
    except (DecompressionBombError, DecompressionBombWarning):
        raise RequestValidationError("IMAGE_TOO_LARGE")
    except UnidentifiedImageError:
        raise RequestValidationError("INVALID_IMAGE")
    except ValueError as e:
        if IMAGE_METADATA_TOO_LARGE_ERROR in str(e):
            raise RequestValidationError("IMAGE_METADATA_TOO_LARGE")
        else:
            raise
    if try_load_image:
        try:
            image.load()
        except (IOError, OSError):
            raise RequestValidationError("INVALID_IMAGE")

    return image


def convert_image_to_np_array(image: PILImage):
    try:
        return np.array(image)
    except (IOError, OSError):
        raise RequestValidationError("INVALID_IMAGE")

def compress_image_to_tensor(image: PILImage):
    to_tensor = transforms.ToTensor()
    return to_tensor(image)

def decompress_image_from_tensor(tensor: torch.Tensor):
    to_pil = transforms.ToPILImage()
    return to_pil(tensor)

def compress_image_to_b64_str(image: PILImage):
    buffer = BytesIO()
    image.save(buffer, "png")
    buffer.seek(0)
    return base64.b64encode(buffer.read()).decode("utf-8")

def get_random_image_b64_str(width=16, height=16, img_format="png"):
    image = Image.fromarray((np.random.rand(width, height, 3) * 255).astype("uint8"))
    buffer = BytesIO()
    image.save(buffer, img_format)
    buffer.seek(0)
    return base64.b64encode(buffer.read()).decode()

