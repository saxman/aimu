from importlib import import_module
from typing import TYPE_CHECKING

# Scorers are pandas-free, so they stay eager and importable from the base package.
from .scorers import LLMJudgeScorer, Scorer

if TYPE_CHECKING:
    from .classification import ClassificationPromptTuner
    from .extraction import ExtractionPromptTuner
    from .judged import JudgedPromptTuner
    from .multiclass import MultiClassPromptTuner

# The tuner classes pull in pandas/tqdm (the `tuning` extra). They are loaded on first
# access so that `import aimu` works without those heavy, optional dependencies.
_LAZY = {
    "ClassificationPromptTuner": ".classification",
    "ExtractionPromptTuner": ".extraction",
    "JudgedPromptTuner": ".judged",
    "MultiClassPromptTuner": ".multiclass",
}


def __getattr__(name: str):
    module_name = _LAZY.get(name)
    if module_name is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    return getattr(import_module(module_name, __name__), name)


__all__ = [
    "ClassificationPromptTuner",
    "ExtractionPromptTuner",
    "JudgedPromptTuner",
    "LLMJudgeScorer",
    "MultiClassPromptTuner",
    "Scorer",
]
