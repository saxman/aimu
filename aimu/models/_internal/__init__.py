"""Private implementation helpers shared across the models package.

Not part of the public API (nothing here is re-exported by ``aimu.models``). These
are cross-cutting utilities used by the base classes (:mod:`aimu.models._base`), the
factories, and/or both the sync and async surfaces:

- ``chat_state``    -- ``_ChatStateMixin`` (system_message / reset / history mechanics)
- ``streaming``     -- stream-chunk filtering (``resolve_include`` / ``filter_chunks``)
- ``json``          -- structured-output extraction from model responses
- ``model_defaults``-- default-model resolution when ``model=`` is omitted
- ``image_input``   -- vision-input normalization (paired with ``image_output``)
- ``image_output``  -- diffusion-output encoding (``encode_image``)
- ``audio_output``  -- audio/speech-output encoding (``encode_audio``)

Provider-specific helpers live with their providers instead (e.g.
``providers/hf/_device.py``, ``providers/_thinking.py``).
"""
