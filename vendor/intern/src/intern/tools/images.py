"""Images for the read tool: an image file becomes an `ImagePart` the provider sends to a model that can see,
resized to a bounded size and stored once; the journal keeps only its path and hash."""

from __future__ import annotations

import hashlib
import io
from pathlib import Path

from PIL import Image

from intern.messages import ImagePart

IMAGE_SUFFIXES = frozenset({".png", ".jpg", ".jpeg", ".gif", ".webp", ".bmp", ".tif", ".tiff"})
# Anthropic's guidance: about 1.15 megapixels and 1568 px on the long edge cost the same as anything
# larger after the server downsizes it. Other vision models accept at least this.
MAX_EDGE = 1568
MAX_PIXELS = 1_150_000


class ImageError(Exception):
    pass


def _fit(width: int, height: int) -> tuple[int, int]:
    scale = min(1.0, MAX_EDGE / max(width, height), (MAX_PIXELS / (width * height)) ** 0.5)
    return max(1, int(width * scale)), max(1, int(height * scale))


def load_image(path: Path, store: Path) -> ImagePart:
    """Open an image (the first frame of an animation), resize it to the bounded size, store it as PNG."""
    try:
        with Image.open(path) as opened:
            image = opened.copy()
    except Exception as e:
        raise ImageError(f"{path.name} could not be opened as an image: {e}") from e
    if image.mode not in ("RGB", "L"):
        background = Image.new("RGB", image.size, "white")
        rgba = image.convert("RGBA")
        background.paste(rgba, mask=rgba.split()[-1])
        image = background
    size = _fit(*image.size)
    if size != image.size:
        image = image.resize(size, Image.Resampling.LANCZOS)
    buffer = io.BytesIO()
    image.save(buffer, format="PNG", optimize=True)
    data = buffer.getvalue()
    digest = hashlib.sha256(data).hexdigest()
    store.mkdir(parents=True, exist_ok=True)
    stored = store / f"{digest[:24]}.png"
    if not stored.exists():
        stored.write_bytes(data)
    return ImagePart(path=str(stored), mime="image/png", width=size[0], height=size[1], source=path.name, sha256=digest)
