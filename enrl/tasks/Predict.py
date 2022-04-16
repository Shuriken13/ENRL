# coding=utf-8

import os
import logging
import pickle
import socket
import datetime
import time
from argparse import ArgumentParser
from pytorch_lightning import seed_everything
from pytorch_lightning.loggers import TensorBoardLogger, CSVLogger

from ..utilities.logging import format_log_metrics_dict
from ..models import *
from ..tasks.Task import Task
from ..configs.settings import *
from ..configs.constants import *
from ..utilities.logging import DEFAULT_LOGGER, metrics_list_best_iter
from ..utilities.io import check_mkdir


class Predict(Task):
    @staticmethod
    def add_task_args(parent_parser):
        parser = Task.add_task_args(parent_parser)
        parser.add_argument('--ckpt_v', type=str, default='',
                            help='If not none, load model from ckpt version, ignore training')
        parser.add_argument('--save_f', type=str, default='',
                            help='Save predict results as pickle file to specific path, default to model checkpoint path')
        return parser

    def __init__(self, ckpt_v, save_f, *args, **kwargs):
        self.ckpt_v = ckpt_v
        self.save_f = save_f
        super().__init__(*args, **kwargs)

    def run(self, *args, **kwargs):
        start_time = time.time()
        self._init_environment()
        model = self._init_model()

        # # init logger
        version = os.path.join(self.model_name, self.ckpt_v) if self.ckpt_v != '' else None
        save_dir, name, version = model.init_logger(version=version)
        self.task_logger = model.model_logger

        # # init trainer
        trainer_args = DEFAULT_TRAINER_ARGS
        if self.trainer_args is not None:
            trainer_args = {**trainer_args, **self.trainer_args}
        model.init_trainer(save_dir=save_dir, name=name, version=version, **trainer_args)

        if self.ckpt_v == '':
            # # train
            model.fit()
        else:
            # # load model
            checkpoint_path = os.path.join(model.log_dir, CKPT_DIR, CKPT_F + '.ckpt')
            hparams_file = os.path.join(model.log_dir, 'hparams.yaml')
            model.load_model(checkpoint_path=checkpoint_path, hparams_file=hparams_file)

        # # test
        predict_result = model.predict(model.get_dataset(phase=PREDICT_PHASE))
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
            'num_para': model.count_variables(),
            'best_iter': best_iter * interval,
            'train_metrics': format_log_metrics_dict(
                model.train_metrics_buf[int(best_iter * interval - 1)]) if len(model.train_metrics_buf) > 0 else '',
            'val_metrics': format_log_metrics_dict(
                model.val_metrics_buf[best_iter]) if len(model.val_metrics_buf) > best_iter else '',
            'test_metrics': format_log_metrics_dict(
                model.test_metrics_buf[-1] if len(model.test_metrics_buf) > 0 else ''),
        }
        self.task_logger.info('')
        for key in task_result:
            self.task_logger.info(key + ': ' + str(task_result[key]))
        save_f = self.save_f if self.save_f != '' else os.path.join(model.log_dir, PREDICT_F)
        check_mkdir(save_f)
        torch.save(predict_result, open(save_f, 'wb'), pickle_protocol=pickle.HIGHEST_PROTOCOL,
                   _use_new_zipfile_serialization=False)
        return predict_result
