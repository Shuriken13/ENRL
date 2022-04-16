# coding=utf-8

import os
import logging
import socket
import datetime
import time
from argparse import ArgumentParser
from pytorch_lightning import seed_everything
from pytorch_lightning.loggers import TensorBoardLogger, CSVLogger

from ..utilities.logging import format_log_metrics_dict
from ..models import *
from ..configs.settings import *
from ..configs.constants import *
from ..utilities.logging import DEFAULT_LOGGER, metrics_list_best_iter


class Task(object):
    @staticmethod
    def add_task_args(parent_parser):
        parser = ArgumentParser(parents=[parent_parser], add_help=False)
        parser.add_argument('--model_name', type=str, default='BiasedMF', help='Choose model to run.')
        parser.add_argument('--dataset', type=str, default='ml100k-5-1', help='Choose dataset to run.')
        parser.add_argument('--random_seed', type=int, default=DEFAULT_SEED,
                            help='Random seed of numpy and pytorch.')
        parser.add_argument('--verbose', type=int, default=logging.INFO,
                            help='Logging Level, 0, 10, ..., 50.')
        # parser.add_argument('--pbar', type=int, default=1,
        #                     help='Whether use tqdm progress bar')
        parser.add_argument('--cuda', type=str, default='0',
                            help='Set CUDA_VISIBLE_DEVICES')
        return parser

    def __init__(self, model_name='BiasedMF', dataset='ml100k-5-1', random_seed=DEFAULT_SEED,
                 verbose=logging.INFO, cuda='0', task_logger=DEFAULT_LOGGER,
                 model_args: dict = None, trainer_args: dict = None, *args, **kwargs):
        self.model_name = model_name
        self.dataset = dataset
        self.random_seed = random_seed
        self.verbose = verbose
        self.cuda = cuda
        self.model_args = model_args
        self.trainer_args = trainer_args
        self.task_logger = task_logger
        # self.pbar = pbar

    def _init_environment(self):
        # cuda
        os.environ["CUDA_VISIBLE_DEVICES"] = self.cuda
        if self.cuda.strip() == '':
            DEFAULT_TRAINER_ARGS['auto_select_gpus'] = 0
            DEFAULT_TRAINER_ARGS['gpus'] = 0
            # Handle create tensor when multiprocessing on some cpu devices, there might be
            # 'RuntimeError: received 0 items of ancdata' in pytorch dataloader
            torch.multiprocessing.set_sharing_strategy('file_system')

        # # init task
        seed_everything(self.random_seed)
        DEFAULT_LOGGER.setLevel(self.verbose)
        self.task_logger.setLevel(self.verbose)
        self.task_logger.info('cuda: {}'.format(self.cuda))
        self.task_logger.info('random_seed: {}'.format(self.random_seed))
        # GLOBAL_ARGS['pbar'] = self.pbar > 0

    def _init_model(self):
        # # init model
        model_name = eval('{0}.{0}'.format(self.model_name))
        model_args = self.model_args if self.model_args is not None else {}
        model = model_name(**model_args)

        # # read data
        reader = model.read_data(dataset_dir=os.path.join(DATASET_DIR, self.dataset))

        # # init modules
        model.init_modules()
        model.summarize(mode='full')

        # # init metrics
        train_metrics = model_args['train_metrics'] if 'train_metrics' in model_args else None
        val_metrics = model_args['val_metrics'] if 'val_metrics' in model_args else None
        test_metrics = model_args['test_metrics'] if 'test_metrics' in model_args else None
        model.init_metrics(train_metrics=train_metrics, val_metrics=val_metrics,
                           test_metrics=test_metrics)
        return model

    def run(self, *args, **kwargs):
        start_time = time.time()
        self._init_environment()
        model = self._init_model()
        num_para = model.count_variables()

        # # init logger
        save_dir, name, version = model.init_logger()
        self.task_logger = model.model_logger

        # # train
        trainer_args = DEFAULT_TRAINER_ARGS
        if self.trainer_args is not None:
            trainer_args = {**trainer_args, **self.trainer_args}
        model.fit(**trainer_args)

        # # test
        test_result = model.test(model.get_dataset(phase=TEST_PHASE))
        end_time = time.time()

        # # format result
        best_iter = metrics_list_best_iter(model.val_metrics_buf)
        interval = trainer_args['val_check_interval']
        task_result = {
            'date': datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            'time': str(datetime.timedelta(seconds=int(end_time - start_time))),
            'server': socket.gethostname(),
            'model_name': self.model_name,
            'version': version.split('/')[-1],
            'num_para': num_para,
            'best_iter': best_iter * interval,
            'train_metrics': format_log_metrics_dict(
                model.train_metrics_buf[int(best_iter * interval - 1)]) if len(model.train_metrics_buf) > 0 else '',
            'val_metrics': format_log_metrics_dict(
                model.val_metrics_buf[best_iter]) if len(model.val_metrics_buf) > best_iter else '',
            'test_metrics': format_log_metrics_dict(
                model.test_metrics_buf[-1] if len(model.test_metrics_buf) > 0 else ''),
        }
        # self.task_logger.info('Task Result: ' + str(task_result))
        self.task_logger.info('')
        for key in task_result:
            self.task_logger.info(key + ': ' + str(task_result[key]))
        return test_result
