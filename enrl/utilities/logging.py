# coding=utf-8
import pytorch_lightning as pl
from pytorch_lightning.loggers import csv_logs
from pytorch_lightning.loggers.base import rank_zero_experiment
import logging
import os
import sys
import hashlib
from tqdm import tqdm
import torch

from ..metrics.metrics import METRICS_SMALLER
from ..configs.constants import *
from ..configs.settings import *

LIGHTNING_LOGGER = logging.getLogger("pytorch_lightning")
for h in LIGHTNING_LOGGER.handlers:
    LIGHTNING_LOGGER.removeHandler(h)
LIGHTNING_LOGGER.addHandler(logging.StreamHandler(sys.stdout))
# DEFAULT_LOGGER = logging.getLogger("enrl")
DEFAULT_LOGGER = LIGHTNING_LOGGER


def create_tqdm(iters, **kwargs):
    default_args = {'ncols': WINDOW_WIDTH, 'mininterval': 1, 'leave': False, 'file': sys.stdout}
    args = {**default_args, **kwargs}
    return tqdm(iters, **args)


def format_log_metrics_dict(metrics: dict):
    log_str = ''
    for key in metrics:
        log_str += ' ' + key
        log_str += '= {:.4f} '.format(metrics[key])
    return log_str.strip()


def format_log_metrics_list(metrics_buf: list):
    metrics = metrics_buf[-1]
    log_str = ''
    for key in metrics:
        if key in METRICS_SMALLER:
            best = min([m[key] for m in metrics_buf])
            better = metrics[key] <= best
        else:
            best = max([m[key] for m in metrics_buf])
            better = metrics[key] >= best
        log_str += ' *' + key if better else ' ' + key
        log_str += '= {:.4f} '.format(metrics[key])
    return log_str


def metrics_list_best_iter(metrics_buf: list):
    best = 0
    for it, metrics in enumerate(metrics_buf):
        better = True
        for key in metrics:
            if (key in METRICS_SMALLER and metrics[key] > metrics_buf[best][key]) or \
                    (key not in METRICS_SMALLER and metrics[key] < metrics_buf[best][key]):
                better = False
                break
            elif (key in METRICS_SMALLER and metrics[key] < metrics_buf[best][key]) or \
                    (key not in METRICS_SMALLER and metrics[key] > metrics_buf[best][key]):
                better = True
                break
        if better:
            best = it
    return best


def logger_add_file_handler(logger, file_path: str, mode='a'):
    for h in logger.handlers:
        if type(h) is logging.FileHandler:
            if h.baseFilename == os.path.abspath(file_path):
                return
    logger.addHandler(logging.FileHandler(filename=file_path, mode=mode))
    return


def hash_hparams(hparams):
    hash_code = hashlib.blake2b(digest_size=10, key=PROJECT_NAME.encode('utf-8'), person=PROJECT_NAME.encode('utf-8'))
    hash_code.update(str(hparams).encode('utf-8'))
    hash_code = hash_code.hexdigest()
    return hash_code


class ModelCheckpoint(pl.callbacks.ModelCheckpoint):
    def _get_metric_interpolated_filepath_name(self, monitor_candidates, trainer, del_filepath):
        return self.format_checkpoint_name(monitor_candidates)


class ExperimentWriter(csv_logs.ExperimentWriter):
    def __init__(self, log_dir: str) -> None:
        self.hparams = {}
        self.metrics = []

        self.log_dir = log_dir
        # if os.path.exists(self.log_dir) and os.listdir(self.log_dir):
        #     rank_zero_warn(
        #         f"Experiment logs directory {self.log_dir} exists and is not empty."
        #         " Previous log files in this directory will be deleted when the new ones are saved!"
        #     )
        os.makedirs(self.log_dir, exist_ok=True)
        self.metrics_file_path = os.path.join(self.log_dir, self.NAME_METRICS_FILE)


class CSVLogger(csv_logs.CSVLogger):
    @property
    @rank_zero_experiment
    def experiment(self) -> ExperimentWriter:
        r"""
        Actual ExperimentWriter object. To use ExperimentWriter features in your
        :class:`~pytorch_lightning.core.lightning.LightningModule` do the following.
        Example::
            self.logger.experiment.some_experiment_writer_function()
        """
        if self._experiment:
            return self._experiment

        os.makedirs(self.root_dir, exist_ok=True)
        self._experiment = ExperimentWriter(log_dir=self.log_dir)
        return self._experiment


class LitProgressBar(pl.callbacks.progress.ProgressBar):

    def init_sanity_tqdm(self) -> tqdm:
        """ Override this to customize the tqdm bar for the validation sanity run. """
        bar = tqdm(
            desc='Validation sanity check',
            position=(2 * self.process_position),
            disable=self.is_disabled,
            leave=False,
            # dynamic_ncols=True,
            ncols=WINDOW_WIDTH,
            file=sys.stdout,
        )
        return bar

    def init_train_tqdm(self) -> tqdm:
        """ Override this to customize the tqdm bar for training. """
        bar = tqdm(
            desc='Training',
            initial=self.train_batch_idx,
            position=(2 * self.process_position),
            disable=self.is_disabled,
            leave=False,
            # dynamic_ncols=True,
            ncols=WINDOW_WIDTH,
            file=sys.stdout,
            smoothing=0,
        )
        return bar

    def init_predict_tqdm(self) -> tqdm:
        """ Override this to customize the tqdm bar for predicting. """
        bar = tqdm(
            desc='Predicting',
            initial=self.train_batch_idx,
            position=(2 * self.process_position),
            disable=self.is_disabled,
            leave=False,
            # dynamic_ncols=True,
            ncols=WINDOW_WIDTH,
            file=sys.stdout,
            smoothing=0,
        )
        return bar

    def init_validation_tqdm(self) -> tqdm:
        """ Override this to customize the tqdm bar for validation. """
        bar = tqdm(
            desc='Validating',
            position=(2 * self.process_position + 1),
            disable=self.is_disabled,
            leave=False,
            # dynamic_ncols=True,
            ncols=WINDOW_WIDTH,
            file=sys.stdout
        )
        return bar

    def init_test_tqdm(self) -> tqdm:
        """ Override this to customize the tqdm bar for testing. """
        bar = tqdm(
            desc="Testing",
            position=(2 * self.process_position),
            disable=self.is_disabled,
            leave=False,
            # dynamic_ncols=True,
            ncols=WINDOW_WIDTH,
            file=sys.stdout
        )
        return bar


class EarlyStopping(pl.callbacks.EarlyStopping):
    def _improvement_message(self, current: torch.Tensor) -> str:
        """ Formats a log message that informs the user about an improvement in the monitored score. """
        if torch.isfinite(self.best_score):
            msg = (
                f"Metric {self.monitor} improved by {abs(self.best_score - current):.4f} >="
                f" min_delta = {abs(self.min_delta)}. New best score: {current:.4f}"
            )
        else:
            msg = f"Metric {self.monitor} improved. New best score: {current:.4f}"
        return msg
