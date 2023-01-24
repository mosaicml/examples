from composer import Callback, Logger, State
from composer.utils import ensure_tuple
from composer.loggers import WandBLogger
from composer.core import Callback, State


class LogDiffusionImages(Callback):

    def eval_after_forward(self, state: State, logger: Logger):
        prompts = state.batch_get_item[0]
        outputs = state.outputs
        for destination in ensure_tuple(logger.destinations):
            if isinstance(destination, WandBLogger):
                destination.log_images(images=outputs, name=prompts, step=state.timestamp.batch.value)