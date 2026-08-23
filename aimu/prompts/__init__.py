from importlib import import_module
from typing import TYPE_CHECKING

# scorers are sqlalchemy-free and pandas-free, so they stay eager.
from .tuners.scorers import LLMJudgeScorer, Scorer

if TYPE_CHECKING:
    from .catalog import Prompt, PromptCatalog
    from .tuner import PromptTuner
    from .tuners import (
        ClassificationPromptTuner,
        ExtractionPromptTuner,
        JudgedPromptTuner,
        MultiClassPromptTuner,
    )

# Prompt/PromptCatalog pull in sqlalchemy (the `prompts` extra); the tuner classes pull
# in pandas/tqdm (the `tuning` extra). Both are loaded on first access so that `import
# aimu` (and `import aimu.agents` / `import aimu.aio`, which reach this package via
# PlanExecuteEvaluator's scorer import) works without those heavy, optional dependencies.
_LAZY = {
    "Prompt": ".catalog",
    "PromptCatalog": ".catalog",
    "PromptTuner": ".tuner",
    "ClassificationPromptTuner": ".tuners",
    "ExtractionPromptTuner": ".tuners",
    "JudgedPromptTuner": ".tuners",
    "MultiClassPromptTuner": ".tuners",
}

if TYPE_CHECKING:  # pragma: no cover
    # Static-analysis-only bindings for names __getattr__ resolves at runtime.
    # PEP 562 lookup is invisible to anything reading the source without importing it,
    # so griffe (behind mkdocstrings) cannot collect these and the docs build aborts on
    # the first one -- being listed in __all__ is not enough, since there is no
    # assignment for a static reader to follow. These imports never execute, so the lazy
    # resolution below still owns runtime behaviour and pandas/tqdm (the [tuning] extra
    # the tuners pull in) stay off `import aimu.prompts`.
    from .catalog import Prompt, PromptCatalog
    from .tuner import PromptTuner
    from .tuners import (
        ClassificationPromptTuner,
        ExtractionPromptTuner,
        JudgedPromptTuner,
        MultiClassPromptTuner,
    )


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
