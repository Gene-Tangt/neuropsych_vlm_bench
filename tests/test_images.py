"""Every image a trial refers to is on disk, is a readable PNG, and is encoded without loss."""
import base64
import struct
import zlib

from neuropsych_bench import load_rows
from neuropsych_bench.dataset import DATA_ROOT
from neuropsych_bench.models import encode_image

PNG_SIGNATURE = b"\x89PNG\r\n\x1a\n"


def image_paths():
    return sorted({DATA_ROOT / p for r in load_rows() for p in r["images"]})


def png_size(data: bytes) -> tuple:
    """Width and height from the IHDR chunk, after checking the signature and the chunk's CRC."""
    assert data[:8] == PNG_SIGNATURE
    length, chunk = struct.unpack(">I4s", data[8:16])
    assert chunk == b"IHDR" and length == 13
    body = data[16:29]
    assert struct.unpack(">I", data[29:33])[0] == zlib.crc32(chunk + body)
    return struct.unpack(">II", body[:8])


def test_images_exist():
    missing = [p for p in image_paths() if not p.is_file()]
    assert not missing, missing[:5]


def test_images_are_valid_png():
    for path in image_paths():
        data = path.read_bytes()
        width, height = png_size(data)
        assert width > 0 and height > 0, path
        assert data.endswith(b"IEND\xaeB`\x82"), f"{path} is truncated"


def test_encode_image_round_trips():
    path = image_paths()[0]
    assert base64.b64decode(encode_image(path)) == path.read_bytes()


if __name__ == "__main__":
    import sys

    import pytest
    sys.exit(pytest.main(["-v", __file__]))
