"""Turns a benchmark row into an ordered list of text and image parts.

This is the only place that decides what a model sees for a trial. Providers (and,
later, framework adapters) translate the parts into their own message format and
must not reorder, add or drop anything.
"""
from dataclasses import dataclass
from pathlib import Path
from typing import Literal, Union

PROMPT_VERSION = "1"  # bump whenever the output of build_parts changes; saved with results

# Text placed before each image in trials that have a target and numbered options
ROLE_CUES = {
    "target": "Here's the target image",
    "option_1": "Here's the first option",
    "option_2": "Here's the second option",
    "option_3": "Here's the third option",
}


@dataclass(frozen=True)
class Text:
    text: str
    type: Literal["text"] = "text"


@dataclass(frozen=True)
class Image:
    path: Path
    role: str
    type: Literal["image"] = "image"


Part = Union[Text, Image]


def uses_cues(roles: list) -> bool:
    """Cue each image when the trial has a target plus numbered options (three- and four-image trials)."""
    return "target" in roles and any(role.startswith("option_") for role in roles)


def build_parts(row: dict, data_root: Path) -> list:
    """The prompt text, then each image in order, each preceded by its cue where the trial uses cues."""
    parts = [Text(row["prompt"])]
    cued = uses_cues(row["image_roles"])
    for path, role in zip(row["images"], row["image_roles"]):
        if cued:
            parts.append(Text(ROLE_CUES[role]))
        parts.append(Image(Path(data_root) / path, role))
    return parts
