"""Image loading / encoding helpers."""

import base64
from io import BytesIO
from PIL import Image
from .config import DEFAULT_IMAGE_MAX_SIZE


def image_to_base64(image_path: str, max_size: int = DEFAULT_IMAGE_MAX_SIZE) -> str:
    """Load image, resize if needed, return base64 string."""
    img = Image.open(image_path).convert("RGB")
    if max(img.size) > max_size:
        img.thumbnail((max_size, max_size))
    buf = BytesIO()
    img.save(buf, format="JPEG", quality=90)
    return base64.b64encode(buf.getvalue()).decode("utf-8")
