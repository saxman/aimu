from importlib import import_module
from typing import TYPE_CHECKING

# catalog and scorers are pandas-free, so they stay eager.
from .catalog import Prompt, PromptCatalog
from .tuners.scorers import LLMJudgeScorer, Scorer

if TYPE_CHECKING:
    from .tuner import PromptTuner
    from .tuners import (
        ClassificationPromptTuner,
        ExtractionPromptTuner,
        JudgedPromptTuner,
        MultiClassPromptTuner,
    )

# The tuner classes pull in pandas/tqdm (the `tuning` extra). They are loaded on first
# access so that `import aimu` works without those heavy, optional dependencies.
_LAZY = {
    "PromptTuner": ".tuner",
    "ClassificationPromptTuner": ".tuners",
    "ExtractionPromptTuner": ".tuners",
    "JudgedPromptTuner": ".tuners",
    "MultiClassPromptTuner": ".tuners",
}


def __getattr__(name: str):
    module_name = _LAZY.get(name)
    if module_name is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    return getattr(import_module(module_name, __name__), name)


__all__ = [
    "Prompt",
    "PromptCatalog",
    "PromptTuner",
    "ClassificationPromptTuner",
    "ExtractionPromptTuner",
    "JudgedPromptTuner",
    "LLMJudgeScorer",
    "MultiClassPromptTuner",
    "Scorer",
]
