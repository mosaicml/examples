from composer.core import Callback, State
from composer.loggers import Logger
from composer.utils import parse_uri, format_name_with_dist_and_time, reproducibility
from composer.loggers.remote_uploader_downloader import RemoteUploaderDownloader
import tempfile
from pathlib import Path
from composer.core.state import fsdp_state_dict_type_context
import torch
import os
from composer.utils import dist


class MonolithicCheckpointSaver(Callback):
    """This callback can be applied upon resuming a model checkpoint.

    Args:
        save_folder (str): Folder to save checkpoints to (can be a URI)
        filename (str): Filename to save checkpoints to.
        batch_interval (int): Number of batches between checkpoints.
    """
    def __init__(self, save_folder: str, batch_interval: int, filename: str='ep{epoch}-ba{batch}-rank{rank}.pt'):
        self.backend, self.bucket_name, self.save_dir_format_str = parse_uri(save_folder)
        self.filename_format_str = filename
        self.batch_interval = batch_interval
        self.upload_to_object_store = (self.backend != '')

    def init(self, state: State, logger: Logger):
        if self.upload_to_object_store:
            remote_ud_exists = False
            remote_uploader_downloaders = [ld for ld in logger.destinations if isinstance(ld, RemoteUploaderDownloader)]
            for remote_ud in remote_uploader_downloaders:
                if remote_ud.remote_backend_name == self.backend and remote_ud.remote_bucket_name == self.bucket_name:
                    remote_ud_exists = True
                    break
            if not remote_ud_exists:
                new_remote_ud = RemoteUploaderDownloader(bucket_uri=f'{self.backend}://{self.bucket_name}')
                new_remote_ud.init(state, logger)
                updated_logger_destinations = [*logger.destinations, new_remote_ud]
                logger.destinations = tuple(updated_logger_destinations)
                state.callbacks.append(new_remote_ud)

    def batch_checkpoint(self, state: State, logger: Logger):
        if state.timestamp.batch.value % self.batch_interval == 0 and dist.get_global_rank() == 0:
            filename = format_name_with_dist_and_time(self.filename_format_str, state.run_name, state.timestamp)
            save_dir = format_name_with_dist_and_time(self.save_dir_format_str, state.run_name, state.timestamp)
            if self.upload_to_object_store:
                with tempfile.TemporaryDirectory() as tempdir:
                    temp_save_path = str(Path(tempdir) / Path(filename))
                    with fsdp_state_dict_type_context(state.model, state_dict_type='full'):
                        state_dict = {'state': {'model': state.model.state_dict()}, 'rng': reproducibility.get_rng_state()}
                        torch.save(state_dict, temp_save_path)
                    remote_file_name = str(Path(save_dir) / Path(filename))
                    logger.upload_file(remote_file_name=remote_file_name, file_path=temp_save_path)
            else:
                save_path = str(Path(save_dir) / Path(filename))
                dirname = os.path.dirname(save_path)
                if dirname:
                    os.makedirs(dirname, exist_ok=True)
                with fsdp_state_dict_type_context(state.model, state_dict_type='full'):
                    state_dict = {'state': {'model': state.model.state_dict()}}
                    torch.save(state_dict, save_path)



