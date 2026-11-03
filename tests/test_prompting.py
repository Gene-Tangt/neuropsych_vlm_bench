from pathlib import Path

import pytest

from neuropsych_bench import load_rows
from neuropsych_bench.dataset import DATA_ROOT
from neuropsych_bench.prompting import ROLE_CUES, Image, Text, build_parts, uses_cues


@pytest.mark.parametrize("roles, cued", [
    (["target"], False),
    (["options", "options"], False),
    (["target", "option_1", "option_2"], True),
    (["target", "option_1", "option_2", "option_3"], True),
])
def test_uses_cues(roles, cued):
    assert uses_cues(roles) is cued


def test_build_parts_cued():
    row = {"prompt": "P", "images": ["a.png", "b.png", "c.png"], "image_roles": ["target", "option_1", "option_2"]}
    assert build_parts(row, Path("root")) == [
        Text("P"),
        Text("Here's the target image"), Image(Path("root/a.png"), "target"),
        Text("Here's the first option"), Image(Path("root/b.png"), "option_1"),
        Text("Here's the second option"), Image(Path("root/c.png"), "option_2"),
    ]


@pytest.mark.parametrize("roles", [["target"], ["options", "options"]])
def test_build_parts_uncued(roles):
    images = [f"{i}.png" for i in range(len(roles))]
    parts = build_parts({"prompt": "P", "images": images, "image_roles": roles}, Path("root"))
    assert parts == [Text("P")] + [Image(Path("root") / p, r) for p, r in zip(images, roles)]


def test_every_trial():
    """Prompt first, then the trial's images in file order, each cued exactly when the layout calls for it."""
    for row in load_rows():
        parts = build_parts(row, DATA_ROOT)
        assert parts[0] == Text(row["prompt"]), row["id"]
        images = [p for p in parts if p.type == "image"]
        assert [p.path for p in images] == [DATA_ROOT / p for p in row["images"]], row["id"]
        assert [p.role for p in images] == row["image_roles"], row["id"]
        if uses_cues(row["image_roles"]):
            assert parts[1:] == [part for role, image in zip(row["image_roles"], images)
                                 for part in (Text(ROLE_CUES[role]), image)], row["id"]
        else:
            assert parts[1:] == images, row["id"]


if __name__ == "__main__":
    import sys

    import pytest
    sys.exit(pytest.main(["-v", __file__]))
