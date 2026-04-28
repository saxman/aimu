try:
    from aimu.evals.deepeval import DeepEvalModel
    from aimu.evals.deepeval_scorer import DeepEvalScorer

    HAS_DEEPEVAL = True
except ImportError:
    HAS_DEEPEVAL = False
    DeepEvalModel = None
    DeepEvalScorer = None

__all__ = ["HAS_DEEPEVAL", "DeepEvalModel", "DeepEvalScorer"]
