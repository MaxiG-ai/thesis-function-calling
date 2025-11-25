import sys

# We will ensure 'ComplexFuncBench' is in sys.path in the main script,
# allowing these imports to work relative to that repo.
try:
    from ComplexFuncBench.runner.gpt_runner import GPTRunner
except ImportError:
    # Fallback for IDE linting, actual execution handled by main script
    pass

from .model import LocalAPIModel


class LocalRunner(GPTRunner):
    def __init__(self, args, logger):
        # Initialize the parent GPTRunner to get all the free logging/state handling
        super().__init__(args, logger)

        # OVERRIDE the model with our Local Proxy
        # We pass the model name defined in your config (e.g., 'llama3-small-local')
        self.model = LocalAPIModel(model_name=args.model_name)
