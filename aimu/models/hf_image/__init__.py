"""HuggingFace ``diffusers``-backed text-to-image client.

Parallel to :mod:`aimu.models.hf` (HuggingFace transformers text client) — same
provider, different modality. Lives under :mod:`aimu.models.hf_image` so the
package name matches the model class (:class:`HuggingFaceImageClient`).
"""

from .hf_image_client import HuggingFaceImageClient, HuggingFaceImageModel

__all__ = ["HuggingFaceImageClient", "HuggingFaceImageModel"]
