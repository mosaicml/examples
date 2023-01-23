from composer import Callback, Event, Logger, State
from composer.utils import ensure_tuple
from composer.loggers import WandBLogger


class LogDiffusionImages(Callback):
    def run_event(self, event: Event, state: State, logger: Logger):
        if event == Event.FIT_END:
            num_images_per_prompt = 1
            prompt = "A pokemon with green eyes, large wings, and a hat"
            images = state.model.module.generate(
                prompt, num_images_per_prompt=num_images_per_prompt)
            for destination in ensure_tuple(logger.destinations):
                if isinstance(destination, WandBLogger):
                    destination.log_images(images=images[0], step=state.timestamp.batch.value)