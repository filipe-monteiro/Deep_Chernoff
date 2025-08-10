"""Utility for plotting Chernoff faces using realistic StyleGAN2 generated faces.

This module loads the pretrained StyleGAN2 generator and several latent space
direction vectors to transform base latent codes according to a set of
numerical attributes.  Each combination of attributes is rendered as a face and
arranged in a grid, in a similar spirit to traditional Chernoff faces but with
photorealistic outputs.

Example
-------
>>> data = [[0.0, 1.0], [-1.0, 0.5]]  # two samples with two attributes
>>> attributes = ["yaw", "age"]
>>> chernoff_chart(data, attributes, out_path="faces.png", weights=None)
The call above creates ``faces.png`` containing two faces.  When ``weights`` is
``None`` the generator uses random weights and therefore produces nonsense
images but the pipeline still executes.  Provide StyleGAN2 weights (e.g.
``'ffhq'``) to obtain realistic faces.
"""

from __future__ import annotations

import math
import os
from typing import List, Mapping, Sequence

import numpy as np
import tensorflow as tf
from matplotlib import pyplot as plt

from stylegan2_generator import StyleGan2Generator
from utils.utils_stylegan2 import convert_images_to_uint8

# Attribute direction files expected in repository root.
_DIRECTION_FILES = [
    "yaw",
    "blinking",
    "joy",
    "age",
    "tone",
    "quality",
    "hairlength",
    "gender",
]


def _load_directions() -> Mapping[str, np.ndarray]:
    """Load latent space direction vectors from ``*.npy`` files."""
    directions = {}
    for name in _DIRECTION_FILES:
        path = os.path.join(os.path.dirname(__file__), f"{name}.npy")
        directions[name] = np.load(path)
    return directions


def _generate_face(
    generator: StyleGan2Generator,
    base_latent: np.ndarray,
    directions: Mapping[str, np.ndarray],
    attributes: Mapping[str, float],
) -> np.ndarray:
    """Return a single face image as ``np.uint8`` array."""
    z = base_latent.copy()
    for key, value in attributes.items():
        if key in directions:
            z = z + directions[key] * float(value)
    images = generator(z)
    images = convert_images_to_uint8(images, nchw_to_nhwc=True).numpy()
    return images[0]


def chernoff_chart(
    data: Sequence[Sequence[float]],
    attributes: Sequence[str],
    out_path: str = "chernoff_chart.png",
    seed: int | None = None,
    weights: str | None = "ffhq",
) -> str:
    """Generate a Chernoff face chart from ``data``.

    Parameters
    ----------
    data:
        Sequence where each item contains the numeric attribute values for one
        observation.  Values are typically in the ``[-1, 1]`` range.
    attributes:
        Names matching the available direction vectors (e.g. ``"age"``,
        ``"yaw"``).
    out_path:
        Location where the resulting grid image will be written.
    seed:
        Random seed controlling the base latent codes.
    weights:
        Name of the StyleGAN2 weight file to load.  ``None`` skips loading
        weights and uses random initialization, which is useful for tests.

    Returns
    -------
    str
        The path where the chart image was saved.
    """
    tf.random.set_seed(seed or 0)
    np.random.seed(seed or 0)

    generator = StyleGan2Generator(weights=weights, gpu=False)
    directions = _load_directions()

    faces: List[np.ndarray] = []
    for row in data:
        attrs = dict(zip(attributes, row))
        base = np.random.randn(1, 512).astype(np.float32)
        faces.append(_generate_face(generator, base, directions, attrs))

    cols = int(math.ceil(math.sqrt(len(faces))))
    rows = int(math.ceil(len(faces) / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 2, rows * 2))
    axes = np.atleast_2d(axes)
    for ax, face in zip(axes.flatten(), faces):
        ax.imshow(face)
        ax.axis("off")
    for ax in axes.flatten()[len(faces):]:
        ax.axis("off")
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)

    return out_path


__all__ = ["chernoff_chart"]
