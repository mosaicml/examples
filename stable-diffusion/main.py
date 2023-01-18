import argparse

import composer
import torch
from model import build_stable_diffusion_model
from data import build_pokemon_datapsec
from composer.utils import dist

from composer import Callback, Event, Logger, State
from composer.callbacks.image_visualizer import _make_input_images
from composer.utils import ensure_tuple
from composer.loggers import WandBLogger

parser = argparse.ArgumentParser()
parser.add_argument('--batch_size', type=int, default=2)
parser.add_argument('--model_name',
                    type=str,
                    default='stabilityai/stable-diffusion-2-base')
parser.add_argument('--image_size', type=int, default=512)
parser.add_argument('--use_ema', action='store_true')
parser.add_argument('--wandb_name', type=str)
parser.add_argument('--wandb_project', type=str)
parser.add_argument('--wandb_group', type=str)
parser.add_argument('--device_train_microbatch_size', type=int)
args = parser.parse_args()


def main(args):
    model = build_stable_diffusion_model(model_name=args.model_name)

    optimizer = torch.optim.AdamW(params=model.parameters(),
                                  lr=1.0e-4,
                                  weight_decay=0.001)
    lr_scheduler = composer.optim.ConstantScheduler()

    train_dataspec = build_pokemon_datapsec(tokenizer=model.tokenizer,
                                            resoltion=args.image_size,
                                            batch_size=args.batch_size //
                                            dist.get_world_size())

    speed_monitor = composer.callbacks.SpeedMonitor(window_size=100)

    logger = composer.loggers.WandBLogger(name=args.wandb_name,
                                          project=args.wandb_project,
                                          group=args.wandb_group)


    device_train_microbatch_size = 'auto'
    if args.device_train_microbatch_size:
        device_train_microbatch_size = args.device_train_microbatch_size

    # callback to visualize images in w&b
    class LogDiffusionImages(Callback):

        def __init__(self, n_imgs):
            self.n_imgs = n_imgs

        def run_event(self, event: Event, state: State, logger: Logger):
            current_time_value = state.eval_timestamp.get('ba').value
            if event == Event.EVAL_BATCH_END and current_time_value == 1:
                images = state.model.generate(n_imgs=self.n_imgs)
                table = _make_input_images(images, self.n_imgs)
                for destination in ensure_tuple(logger.destinations):
                    if isinstance(destination, WandBLogger):
                        destination.log_metrics({'Image': table}, state.timestamp.batch.value)

    log_images = LogDiffusionImages(n_imgs=4)

    trainer = composer.Trainer(
        model=model,
        train_dataloader=train_dataspec,
        optimizers=optimizer,
        schedulers=lr_scheduler,
        callbacks=[speed_monitor, log_images],
        loggers=logger,
        max_duration='5ep',
        device_train_microbatch_size=device_train_microbatch_size,
    )
    trainer.fit()


if __name__ == "__main__":
    print(args)
    main(args)