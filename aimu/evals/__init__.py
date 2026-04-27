try:
    from aimu.evals.deepeval import DeepEvalModel

    HAS_DEEPEVAL = True
except ImportError:
    HAS_DEEPEVAL = False
