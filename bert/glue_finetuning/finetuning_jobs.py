# # Copyright 2022 MosaicML Composer authors
# # SPDX-License-Identifier: Apache-2.0

# """Contains GLUE job objects for the simple_glue_trainer."""
import composer  # type: ignore
import multiprocessing as mp
from typing import Any, Dict, List, Optional, Union, cast

import torch  # type: ignore
from torch.utils.data import DataLoader

from composer.core import Callback
from composer.core.evaluator import Evaluator
from composer.core.types import Dataset
from composer.loggers import LoggerDestination
from composer.models.bert.model import create_bert_classification
from composer.optim import ComposerScheduler, LinearWithWarmupScheduler, DecoupledAdamW
from composer.trainer.devices import Device, DeviceGPU
from composer.trainer.trainer import Trainer
from composer.utils import dist, reproducibility

from data import create_glue_dataset
from model import create_bert_for_glue


def _build_dataloader(dataset, **kwargs):
    import transformers
    dataset = cast(Dataset, dataset)

    return DataLoader(
        dataset=dataset,
        sampler=dist.get_sampler(dataset, drop_last=False, shuffle=False),
        collate_fn=transformers.default_data_collator,
        **kwargs,
    )


Metrics = Dict[str, Dict[str, Any]]


class FineTuneJob:
    """Encapsulates a fine-tuning job.

    Tasks should subclass FineTuneJob and implement the
    get_trainer() method.

    Args:
        name (str, optional): job name. Defaults to the class name.
        load_path (str, optional): path to load checkpoints. Default: None
        save_folder (str, optional): path to save checkpoints. Default: None
        kwargs (dict, optional): additional arguments passed available to the Trainer.
    """

    def __init__(
        self,
        job_name: Optional[str] = None,
        load_path: Optional[str] = None,
        save_folder: Optional[str] = None,
        seed: int = 42,
        **kwargs,
    ):
        reproducibility.seed_all(seed)
        self._job_name = job_name
        self.seed = seed
        self.load_path = load_path
        self.save_folder = save_folder
        self.kwargs = kwargs

    def get_trainer(self, device: Optional[Union[str, Device]]) -> Trainer:
        """Returns the trainer for the job."""
        raise NotImplementedError

    def print_metrics(self, metrics: Metrics):
        """Prints fine-tuning results."""
        job_name = self.job_name

        print(f'Results for {job_name}:')
        print('-' * (12 + len(job_name)))
        for eval, metric in metrics.items():
            for metric_name, value in metric.items():
                print(f'{eval}: {metric_name}, {value*100:.2f}')
        print('-' * (12 + len(job_name)))

    @property
    def job_name(self) -> str:
        """Job name, defaults to class name."""
        if self._job_name is not None:
            return self._job_name
        return self.__class__.__name__

    def run(self, gpu_queue: Optional[mp.Queue] = None) -> Dict[str, Any]:
        """Trains the model, optionally pulling a GPU id from the queue.

        Returns a dict with keys:
        * 'checkpoints': list of saved_checkpoints, if any,
        * 'metrics': nested dict of results, accessed by
                     dataset and metric name, e.g.
                     ``metrics['glue_mnli']['Accuracy']``.
        """
        gpu_id = gpu_queue.get() if gpu_queue is not None else 0

        print(f'Running {self.job_name} on GPU {gpu_id}')

        try:
            gpu = DeviceGPU(gpu_id)
            trainer = self.get_trainer(device=gpu)

            trainer.fit()

            collected_metrics: Dict[str, Dict[str, Any]] = {}
            for eval_name, metrics in trainer.state.eval_metrics.items():
                collected_metrics[eval_name] = {
                    name: metric.compute().cpu().numpy() for name, metric in metrics.items()
                }

            trainer.close()
            self.print_metrics(collected_metrics)
        finally:
            # release the GPU for other jobs
            if gpu_queue:
                print(f'Releasing GPU {gpu_id}')
                gpu_queue.put(gpu_id)

        return {'checkpoints': trainer.saved_checkpoints, 'metrics': collected_metrics, 'job_name': self.job_name}


class GlueClassificationJob(FineTuneJob):

    def __init__(
        self,
        job_name: Optional[str] = None,
        seed: int = 42,
        task_name: Optional[str] = None,
        num_labels: Optional[int] = -1,
        eval_interval: str = '1000ba',
        pretrained_model_name: str = 'bert-base-uncased',
        tokenizer_name: Optional[str] = None,
        scheduler: Optional[ComposerScheduler] = None,
        max_sequence_length: Optional[int] = 256,
        max_duration: Optional[str] = '3ep',
        batch_size: Optional[int] = 32,
        load_path: Optional[str] = None,
        save_folder: Optional[str] = None,
        loggers: Optional[List[LoggerDestination]] = None,
        callbacks: Optional[List[Callback]] = None,
        precision: Optional[str] = None,
        **kwargs,
    ):
        if task_name is None:
            raise ValueError(
                "GlueClassificationJob should not be instantiated directly. Please instantiate a specific glue job type instead (e.g. MNLIJob)."
            )
        super().__init__(job_name, load_path, save_folder, seed, **kwargs)

        self.task_name = task_name

        self.num_labels = num_labels
        self.eval_interval = eval_interval
        self.pretrained_model_name = pretrained_model_name
        self.tokenizer_name = tokenizer_name if tokenizer_name is not None else pretrained_model_name

        self.model = create_bert_for_glue(num_labels=self.num_labels,
                                          pretrained_model_name=self.pretrained_model_name,
                                          use_pretrained=True)

        self.scheduler = scheduler

        self.max_sequence_length = max_sequence_length
        self.max_duration = max_duration
        self.batch_size = batch_size
        self.loggers = loggers
        self.callbacks = callbacks
        self.precision = precision

        # These will be set by the subclasses for specific GLUE tasks
        self.train_dataloader = None
        self.evaluators = None
        self.optimizer = None

    def get_trainer(self, device: Optional[Union[Device, str]] = None):
        return Trainer(model=self.model,
                       optimizers=self.optimizer,
                       schedulers=self.scheduler,
                       train_dataloader=self.train_dataloader,
                       eval_dataloader=self.evaluators,
                       eval_interval=self.eval_interval,
                       load_path=self.load_path,
                       save_folder=self.save_folder,
                       max_duration=self.max_duration,
                       seed=self.seed,
                       grad_accum='auto',
                       load_weights_only=True,
                       load_strict_model_weights=False,
                       loggers=self.loggers,
                       callbacks=self.callbacks,
                       python_log_level='ERROR',
                       run_name=self.job_name,
                       load_ignore_keys=['state/model/model.classifier*'],
                       precision=self.precision,
                       device=device,
                       progress_bar=True,
                       log_to_console=False,
                       **self.kwargs)


class MNLIJob(GlueClassificationJob):
    """MNLI."""

    def __init__(
        self,
        job_name: Optional[str] = None,
        seed: int = 42,
        eval_interval: str = '2300ba',
        pretrained_model_name: str = 'bert-base-uncased',
        tokenizer_name: Optional[str] = None,
        scheduler: Optional[ComposerScheduler] = None,
        max_sequence_length: Optional[int] = 256,
        max_duration: Optional[str] = '3ep',
        batch_size: Optional[int] = 48,
        load_path: Optional[str] = None,
        save_folder: Optional[str] = None,
        loggers: Optional[List[LoggerDestination]] = None,
        callbacks: Optional[List[Callback]] = None,
        precision: Optional[str] = None,
        **kwargs,
    ):
        super().__init__(job_name=job_name,
                         seed=seed,
                         task_name='mnli',
                         num_labels=3,
                         eval_interval=eval_interval,
                         pretrained_model_name=pretrained_model_name,
                         tokenizer_name=tokenizer_name,
                         scheduler=scheduler,
                         max_sequence_length=max_sequence_length,
                         max_duration=max_duration,
                         batch_size=batch_size,
                         load_path=load_path,
                         save_folder=save_folder,
                         loggers=loggers,
                         callbacks=callbacks,
                         precision=precision,
                         **kwargs)

        self.optimizer = DecoupledAdamW(self.model.parameters(),
                                        lr=5.0e-5,
                                        betas=[0.9, 0.98],
                                        eps=1.0e-06,
                                        weight_decay=5.0e-06)

        dataset_kwargs = {
            'task': self.task_name,
            'tokenizer_name': tokenizer_name,
            'max_seq_length': max_sequence_length,
        }

        dataloader_kwargs = {
            'batch_size': batch_size,
            'num_workers': 0,
            'shuffle': False,
            'drop_last': False,
        }
        train_dataset = create_glue_dataset(split='train', **dataset_kwargs)
        self.train_dataloader = _build_dataloader(train_dataset, **dataloader_kwargs)
        mnli_eval_dataset = create_glue_dataset(split='validation_matched', **dataset_kwargs)
        mnli_eval_mismatched_dataset = create_glue_dataset(split='validation_mismatched', **dataset_kwargs)
        mnli_evaluator = Evaluator(label='glue_mnli',
                                   dataloader=_build_dataloader(mnli_eval_dataset, **dataloader_kwargs),
                                   metric_names=['Accuracy'])
        mnli_evaluator_mismatched = Evaluator(label='glue_mnli_mismatched',
                                              dataloader=_build_dataloader(mnli_eval_mismatched_dataset,
                                                                           **dataloader_kwargs),
                                              metric_names=['Accuracy'])
        self.evaluators = [mnli_evaluator, mnli_evaluator_mismatched]


class RTEJob(GlueClassificationJob):
    """RTE."""

    def __init__(
        self,
        job_name: Optional[str] = None,
        seed: int = 42,
        eval_interval: str = '1000ba',
        pretrained_model_name: str = 'bert-base-uncased',
        tokenizer_name: Optional[str] = None,
        scheduler: Optional[ComposerScheduler] = None,
        max_sequence_length: Optional[int] = 256,
        max_duration: Optional[str] = '3ep',
        batch_size: Optional[int] = 32,
        load_path: Optional[str] = None,
        save_folder: Optional[str] = None,
        loggers: Optional[List[LoggerDestination]] = None,
        callbacks: Optional[List[Callback]] = None,
        precision: Optional[str] = None,
        **kwargs,
    ):
        super().__init__(job_name=job_name,
                         seed=seed,
                         task_name='rte',
                         num_labels=2,
                         eval_interval=eval_interval,
                         pretrained_model_name=pretrained_model_name,
                         tokenizer_name=tokenizer_name,
                         scheduler=scheduler,
                         max_sequence_length=max_sequence_length,
                         max_duration=max_duration,
                         batch_size=batch_size,
                         load_path=load_path,
                         save_folder=save_folder,
                         loggers=loggers,
                         callbacks=callbacks,
                         precision=precision,
                         **kwargs)

        self.optimizer = DecoupledAdamW(self.model.parameters(),
                                        lr=1.0e-5,
                                        betas=[0.9, 0.98],
                                        eps=1.0e-06,
                                        weight_decay=1.0e-5)

        dataset_kwargs = {
            'task': self.task_name,
            'tokenizer_name': tokenizer_name,
            'max_seq_length': max_sequence_length,
        }

        dataloader_kwargs = {
            'batch_size': batch_size,
            'num_workers': 0,
            'shuffle': False,
            'drop_last': False,
        }
        train_dataset = create_glue_dataset(split='train', **dataset_kwargs)
        self.train_dataloader = _build_dataloader(train_dataset, **dataloader_kwargs)
        rte_eval_dataset = create_glue_dataset(split='validation', **dataset_kwargs)
        rte_evaluator = Evaluator(label='glue_rte',
                                  dataloader=_build_dataloader(rte_eval_dataset, **dataloader_kwargs),
                                  metric_names=['Accuracy'])
        self.evaluators = [rte_evaluator]


class QQPJob(GlueClassificationJob):
    """QQP."""

    def __init__(
        self,
        job_name: Optional[str] = None,
        seed: int = 42,
        eval_interval: str = '2000ba',
        pretrained_model_name: str = 'bert-base-uncased',
        tokenizer_name: Optional[str] = None,
        scheduler: Optional[ComposerScheduler] = None,
        max_sequence_length: Optional[int] = 256,
        max_duration: Optional[str] = '5ep',
        batch_size: Optional[int] = 16,
        load_path: Optional[str] = None,
        save_folder: Optional[str] = None,
        loggers: Optional[List[LoggerDestination]] = None,
        callbacks: Optional[List[Callback]] = None,
        precision: Optional[str] = None,
        **kwargs,
    ):
        if optimizer_kwargs is None:
            optimizer_kwargs = {'lr': 3.0e-5, 'betas': [0.9, 0.98], 'eps': 1.0e-06, 'weight_decay': 3.0e-6}
        super().__init__(job_name=job_name,
                         seed=seed,
                         task_name='qqp',
                         num_labels=2,
                         eval_interval=eval_interval,
                         pretrained_model_name=pretrained_model_name,
                         tokenizer_name=tokenizer_name,
                         scheduler=scheduler,
                         max_sequence_length=max_sequence_length,
                         max_duration=max_duration,
                         batch_size=batch_size,
                         load_path=load_path,
                         save_folder=save_folder,
                         loggers=loggers,
                         callbacks=callbacks,
                         precision=precision,
                         **kwargs)

        self.optimizer = DecoupledAdamW(self.model.parameters(),
                                        lr=3.0e-5,
                                        betas=[0.9, 0.98],
                                        eps=1.0e-06,
                                        weight_decay=3.0e-6)

        dataset_kwargs = {
            'task': self.task_name,
            'tokenizer_name': tokenizer_name,
            'max_seq_length': max_sequence_length,
        }

        dataloader_kwargs = {
            'batch_size': batch_size,
            'num_workers': 0,
            'shuffle': False,
            'drop_last': False,
        }
        train_dataset = create_glue_dataset(split='train', **dataset_kwargs)
        self.train_dataloader = _build_dataloader(train_dataset, **dataloader_kwargs)
        qqp_eval_dataset = create_glue_dataset(split='validation', **dataset_kwargs)
        qqp_evaluator = Evaluator(label='glue_qqp',
                                  dataloader=_build_dataloader(qqp_eval_dataset, **dataloader_kwargs),
                                  metric_names=['Accuracy', 'BinaryF1Score'])
        self.evaluators = [qqp_evaluator]


class COLAJob(GlueClassificationJob):
    """COLA."""

    def __init__(
        self,
        job_name: Optional[str] = None,
        seed: int = 42,
        eval_interval: str = '250ba',
        pretrained_model_name: str = 'bert-base-uncased',
        tokenizer_name: Optional[str] = None,
        scheduler: Optional[ComposerScheduler] = None,
        max_sequence_length: Optional[int] = 256,
        max_duration: Optional[str] = '10ep',
        batch_size: Optional[int] = 32,
        load_path: Optional[str] = None,
        save_folder: Optional[str] = None,
        loggers: Optional[List[LoggerDestination]] = None,
        callbacks: Optional[List[Callback]] = None,
        precision: Optional[str] = None,
        **kwargs,
    ):
        if optimizer_kwargs is None:
            optimizer_kwargs = {'lr': 5.0e-5, 'betas': [0.9, 0.98], 'eps': 1.0e-06, 'weight_decay': 5.0e-6}
        super().__init__(job_name=job_name,
                         seed=seed,
                         task_name='cola',
                         num_labels=2,
                         eval_interval=eval_interval,
                         pretrained_model_name=pretrained_model_name,
                         tokenizer_name=tokenizer_name,
                         scheduler=scheduler,
                         max_sequence_length=max_sequence_length,
                         max_duration=max_duration,
                         batch_size=batch_size,
                         load_path=load_path,
                         save_folder=save_folder,
                         loggers=loggers,
                         callbacks=callbacks,
                         precision=precision,
                         **kwargs)

        self.optimizer = DecoupledAdamW(self.model.parameters(),
                                        lr=5.0e-5,
                                        betas=[0.9, 0.98],
                                        eps=1.0e-06,
                                        weight_decay=5.0e-6)

        dataset_kwargs = {
            'task': self.task_name,
            'tokenizer_name': tokenizer_name,
            'max_seq_length': max_sequence_length,
        }

        dataloader_kwargs = {
            'batch_size': batch_size,
            'num_workers': 0,
            'shuffle': False,
            'drop_last': False,
        }
        train_dataset = create_glue_dataset(split='train', **dataset_kwargs)
        self.train_dataloader = _build_dataloader(train_dataset, **dataloader_kwargs)
        cola_eval_dataset = create_glue_dataset(split='validation', **dataset_kwargs)
        cola_evaluator = Evaluator(label='glue_cola',
                                   dataloader=_build_dataloader(cola_eval_dataset, **dataloader_kwargs),
                                   metric_names=['MatthewsCorrCoef'])
        self.evaluators = [cola_evaluator]


class MRPCJob(GlueClassificationJob):
    """MRPC."""

    def __init__(
        self,
        job_name: Optional[str] = None,
        seed: int = 42,
        eval_interval: str = '100ba',
        pretrained_model_name: str = 'bert-base-uncased',
        tokenizer_name: Optional[str] = None,
        scheduler: Optional[ComposerScheduler] = None,
        max_sequence_length: Optional[int] = 256,
        max_duration: Optional[str] = '10ep',
        batch_size: Optional[int] = 32,
        load_path: Optional[str] = None,
        save_folder: Optional[str] = None,
        loggers: Optional[List[LoggerDestination]] = None,
        callbacks: Optional[List[Callback]] = None,
        precision: Optional[str] = None,
        **kwargs,
    ):
        super().__init__(job_name=job_name,
                         seed=seed,
                         task_name='mrpc',
                         num_labels=2,
                         eval_interval=eval_interval,
                         pretrained_model_name=pretrained_model_name,
                         tokenizer_name=tokenizer_name,
                         scheduler=scheduler,
                         max_sequence_length=max_sequence_length,
                         max_duration=max_duration,
                         batch_size=batch_size,
                         load_path=load_path,
                         save_folder=save_folder,
                         loggers=loggers,
                         callbacks=callbacks,
                         precision=precision,
                         **kwargs)

        self.optimizer = DecoupledAdamW(self.model.parameters(),
                                        lr=8.0e-5,
                                        betas=[0.9, 0.98],
                                        eps=1.0e-06,
                                        weight_decay=8.0e-6)

        dataset_kwargs = {
            'task': self.task_name,
            'tokenizer_name': tokenizer_name,
            'max_seq_length': max_sequence_length,
        }

        dataloader_kwargs = {
            'batch_size': batch_size,
            'num_workers': 0,
            'shuffle': False,
            'drop_last': False,
        }
        train_dataset = create_glue_dataset(split='train', **dataset_kwargs)
        self.train_dataloader = _build_dataloader(train_dataset, **dataloader_kwargs)
        mrpc_eval_dataset = create_glue_dataset(split='validation', **dataset_kwargs)
        mrpc_evaluator = Evaluator(label='glue_mrpc',
                                   dataloader=_build_dataloader(mrpc_eval_dataset, **dataloader_kwargs),
                                   metric_names=['Accuracy', 'BinaryF1Score'])
        self.evaluators = [mrpc_evaluator]


class QNLIJob(GlueClassificationJob):
    """QNLI."""

    def __init__(
        self,
        job_name: Optional[str] = None,
        seed: int = 42,
        eval_interval: str = '1000ba',
        pretrained_model_name: str = 'bert-base-uncased',
        tokenizer_name: Optional[str] = None,
        scheduler: Optional[ComposerScheduler] = None,
        max_sequence_length: Optional[int] = 256,
        max_duration: Optional[str] = '10ep',
        batch_size: Optional[int] = 16,
        load_path: Optional[str] = None,
        save_folder: Optional[str] = None,
        loggers: Optional[List[LoggerDestination]] = None,
        callbacks: Optional[List[Callback]] = None,
        precision: Optional[str] = None,
        **kwargs,
    ):
        super().__init__(job_name=job_name,
                         seed=seed,
                         task_name='qnli',
                         num_labels=2,
                         eval_interval=eval_interval,
                         pretrained_model_name=pretrained_model_name,
                         tokenizer_name=tokenizer_name,
                         scheduler=scheduler,
                         max_sequence_length=max_sequence_length,
                         max_duration=max_duration,
                         batch_size=batch_size,
                         load_path=load_path,
                         save_folder=save_folder,
                         loggers=loggers,
                         callbacks=callbacks,
                         precision=precision,
                         **kwargs)

        self.optimizer = DecoupledAdamW(self.model.parameters(),
                                        lr=1.0e-5,
                                        betas=[0.9, 0.98],
                                        eps=1.0e-06,
                                        weight_decay=1.0e-6)

        dataset_kwargs = {
            'task': self.task_name,
            'tokenizer_name': tokenizer_name,
            'max_seq_length': max_sequence_length,
        }

        dataloader_kwargs = {
            'batch_size': batch_size,
            'num_workers': 0,
            'shuffle': False,
            'drop_last': False,
        }
        train_dataset = create_glue_dataset(split='train', **dataset_kwargs)
        self.train_dataloader = _build_dataloader(train_dataset, **dataloader_kwargs)
        qnli_eval_dataset = create_glue_dataset(split='validation', **dataset_kwargs)
        qnli_evaluator = Evaluator(label='glue_qnli',
                                   dataloader=_build_dataloader(qnli_eval_dataset, **dataloader_kwargs),
                                   metric_names=['Accuracy'])
        self.evaluators = [qnli_evaluator]


class SST2Job(GlueClassificationJob):
    """SST2."""

    def __init__(
        self,
        job_name: Optional[str] = None,
        seed: int = 42,
        eval_interval: str = '500ba',
        pretrained_model_name: str = 'bert-base-uncased',
        tokenizer_name: Optional[str] = None,
        scheduler: Optional[ComposerScheduler] = None,
        max_sequence_length: Optional[int] = 256,
        max_duration: Optional[str] = '3ep',
        batch_size: Optional[int] = 16,
        load_path: Optional[str] = None,
        save_folder: Optional[str] = None,
        loggers: Optional[List[LoggerDestination]] = None,
        callbacks: Optional[List[Callback]] = None,
        precision: Optional[str] = None,
        **kwargs,
    ):
        super().__init__(job_name=job_name,
                         seed=seed,
                         task_name='sst2',
                         num_labels=2,
                         eval_interval=eval_interval,
                         pretrained_model_name=pretrained_model_name,
                         tokenizer_name=tokenizer_name,
                         scheduler=scheduler,
                         max_sequence_length=max_sequence_length,
                         max_duration=max_duration,
                         batch_size=batch_size,
                         load_path=load_path,
                         save_folder=save_folder,
                         loggers=loggers,
                         callbacks=callbacks,
                         precision=precision,
                         **kwargs)

        self.optimizer = DecoupledAdamW(self.model.parameters(),
                                        lr=3.0e-5,
                                        betas=[0.9, 0.98],
                                        eps=1.0e-06,
                                        weight_decay=3.0e-6)

        dataset_kwargs = {
            'task': self.task_name,
            'tokenizer_name': tokenizer_name,
            'max_seq_length': max_sequence_length,
        }

        dataloader_kwargs = {
            'batch_size': batch_size,
            'num_workers': 0,
            'shuffle': False,
            'drop_last': False,
        }
        train_dataset = create_glue_dataset(split='train', **dataset_kwargs)
        self.train_dataloader = _build_dataloader(train_dataset, **dataloader_kwargs)
        sst2_eval_dataset = create_glue_dataset(split='validation', **dataset_kwargs)
        sst2_evaluator = Evaluator(label='glue_sst2',
                                   dataloader=_build_dataloader(sst2_eval_dataset, **dataloader_kwargs),
                                   metric_names=['Accuracy'])
        self.evaluators = [sst2_evaluator]


class STSBJob(GlueClassificationJob):
    """STSB."""

    def __init__(
        self,
        job_name: Optional[str] = None,
        seed: int = 42,
        eval_interval: str = '2000ba',
        pretrained_model_name: str = 'bert-base-uncased',
        tokenizer_name: Optional[str] = None,
        scheduler: Optional[ComposerScheduler] = None,
        max_sequence_length: Optional[int] = 256,
        max_duration: Optional[str] = '10ep',
        batch_size: Optional[int] = 32,
        load_path: Optional[str] = None,
        save_folder: Optional[str] = None,
        loggers: Optional[List[LoggerDestination]] = None,
        callbacks: Optional[List[Callback]] = None,
        precision: Optional[str] = None,
        **kwargs,
    ):
        super().__init__(job_name=job_name,
                         seed=seed,
                         task_name='stsb',
                         num_labels=1,
                         eval_interval=eval_interval,
                         pretrained_model_name=pretrained_model_name,
                         tokenizer_name=tokenizer_name,
                         scheduler=scheduler,
                         max_sequence_length=max_sequence_length,
                         max_duration=max_duration,
                         batch_size=batch_size,
                         load_path=load_path,
                         save_folder=save_folder,
                         loggers=loggers,
                         callbacks=callbacks,
                         precision=precision,
                         **kwargs)

        self.optimizer = DecoupledAdamW(self.model.parameters(),
                                        lr=3.0e-5,
                                        betas=[0.9, 0.98],
                                        eps=1.0e-06,
                                        weight_decay=3.0e-6)

        dataset_kwargs = {
            'task': self.task_name,
            'tokenizer_name': tokenizer_name,
            'max_seq_length': max_sequence_length,
        }

        dataloader_kwargs = {
            'batch_size': batch_size,
            'num_workers': 0,
            'shuffle': False,
            'drop_last': False,
        }
        train_dataset = create_glue_dataset(split='train', **dataset_kwargs)
        self.train_dataloader = _build_dataloader(train_dataset, **dataloader_kwargs)
        stsb_eval_dataset = create_glue_dataset(split='validation', **dataset_kwargs)
        stsb_evaluator = Evaluator(label='glue_stsb',
                                   dataloader=_build_dataloader(stsb_eval_dataset, **dataloader_kwargs),
                                   metric_names=['SpearmanCorrCoef'])
        self.evaluators = [stsb_evaluator]

        # Hardcoded for STSB due to a bug (Can be removed once torchmetrics fixes https://github.com/Lightning-AI/metrics/issues/1294)
        self.precision = 'fp32'
