"""Google Gemini image-generation client (Nano Banana etc.).

Parallel to the text Gemini client at :mod:`aimu.models.openai_compat.gemini_client`
— same provider, different modality, different SDK (uses the native ``google-genai``
package rather than the OpenAI-compat endpoint).
"""

from .gemini_image_client import GeminiImageClient, GeminiImageModel

__all__ = ["GeminiImageClient", "GeminiImageModel"]
