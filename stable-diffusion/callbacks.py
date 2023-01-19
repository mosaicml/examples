from composer import Callback, Event, Logger, State
from composer.callbacks.image_visualizer import _make_input_images
from composer.utils import ensure_tuple
from composer.loggers import WandBLogger


class LogDiffusionImages(Callback):
    def run_event(self, event: Event, state: State, logger: Logger):
        if event == Event.FIT_END:
            num_images_per_prompt = 1
            prompt = "A pokemon with green eyes, large wings, and a hat"
            images = state.model.generate(
                prompt, num_images_per_prompt=num_images_per_prompt)
            table = _make_input_images(images, num_images_per_prompt)
            for destination in ensure_tuple(logger.destinations):
                if isinstance(destination, WandBLogger):
                    destination.log_metrics({'Image': table},
                                            state.timestamp.batch.value)