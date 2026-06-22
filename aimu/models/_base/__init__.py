"""Per-modality base types, split out of :mod:`aimu.models.base`.

The public import location stays ``aimu.models.base`` (a thin re-export hub). These
submodules group the four independent modality stacks so each reads top-to-bottom:

- :mod:`.shared`: ``StreamChunk`` / ``StreamingContentType`` / ``classproperty`` (cross-modality)
- :mod:`.text`: ``ModelSpec`` / ``Model`` / ``BaseModelClient``
- :mod:`.image`: image specs / ``ImageModel`` / ``BaseImageClient``
- :mod:`.audio`: audio specs / ``AudioModel`` / ``BaseAudioClient``
- :mod:`.speech`: speech specs / ``SpeechModel`` / ``BaseSpeechClient``
"""
