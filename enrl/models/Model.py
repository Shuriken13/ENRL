# coding=utf-8
import torch
import numpy as np
import pytorch_lightning as pl
from pytorch_lightning.callbacks import EarlyStopping
# from pytorch_lightning.loggers import TensorBoardLogger, CSVLogger, TestTubeLogger
from argparse import ArgumentParser, Namespace
import copy
import inspect
import os
import hashlib
import yaml
import logging
import sys

from ..datasets import *
from ..datasets import Dataset
from ..data_readers import *
from ..configs.constants import *
from ..configs.settings import *
from ..metrics.MetricList import MetricsList
from ..metrics.metrics import METRICS_SMALLER
from ..utilities.logging import *
from ..utilities.io import check_mkdir
from ..utilities.argument import get_class_init_args


class Model(pl.LightningModule):
    default_reader = 'DataReader'
    default_dataset = 'Dataset'

    @staticmethod
    def add_model_specific_args(parent_parser):
        """
        模型命令行参数
        :param parser:
        :param model_name: 模型名称
        :return:
        """
        parser = ArgumentParser(parents=[parent_parser], add_help=False)
        parser.add_argument('--lr', type=float, default=0.001,
                            help='Learning rate.')
        parser.add_argument('--optimizer_name', type=str, default='Adam',
                            help='optimizer: GD, Adam, Adagrad')
        parser.add_argument('--dropout', type=float, default=0.0,
                            help='Dropout probability for each deep layer')
        parser.add_argument('--l2_bias', type=int, default=0,
                            help='Whether add l2 regularizer on bias.')
        parser.add_argument('--l2', type=float, default=1e-6,
                            help='Weight of l2_regularize in pytorch optimizer.')
        parser.add_argument('--loss_sum', type=int, default=1,
                            help='Reduction of batch loss 1=sum, 0=mean')
        parser.add_argument('--loss_type', type=str, default='bce',
                            help='Type of loss function, such as bpr, mse, ce')
        parser.add_argument('--buffer_ds', type=int, default=0,
                            help='Whether buffer dataset items or not.')
        parser.add_argument('--batch_size', type=int, default=128,
                            help='Batch size during training.')
        parser.add_argument('--eval_batch_size', type=int, default=16,
                            help='Batch size during testing.')
        parser.add_argument('--num_workers', type=int, default=2,
                            help='Number of processors when get batches in DataLoader')
        parser.add_argument('--es_patience', type=int, default=50,
                            help='#epochs with no improvement after which training will be stopped (early stop).')
        parser.add_argument('--train_metrics', type=str, default='',
                            help='Calculate metrics on training')
        parser.add_argument('--val_metrics', type=str, default='auc',
                            help='Calculate metrics on validation')
        parser.add_argument('--test_metrics', type=str, default='auc',
                            help='Calculate metrics on testing')
        return parser

    def __init__(self,
                 lr: float = 0.001, optimizer_name: str = 'Adam', dropout: float = 0.2,
                 l2: float = 1e-6, l2_bias: int = 0, loss_sum: int = 1, loss_type: str = 'mse',
                 batch_size: int = 128, eval_batch_size: int = 16, es_patience: int = 20,
                 buffer_ds: int = 0, num_workers: int = 4,
                 *args, **kwargs):
        super().__init__()
        self.lr = lr
        self.optimizer_name = optimizer_name
        self.dropout = dropout
        self.l2 = l2
        self.l2_bias = l2_bias
        self.loss_sum = loss_sum
        self.loss_type = loss_type
        self.buffer_ds = buffer_ds
        self.batch_size = batch_size
        self.eval_batch_size = eval_batch_size
        self.num_workers = num_workers
        self.es_patience = es_patience

        self.train_metrics = None
        self.train_metrics_buf = []
        self.val_metrics = None
        self.val_metrics_buf = []
        self.test_metrics = None
        self.test_metrics_buf = []
        self.reader = None
        self.train_dataset = None
        self.val_dataset = None
        self.test_dataset = None

        self.model_logger = DEFAULT_LOGGER
        self.log_dir = None

    def save_hyperparameters(self, *args, frame=None) -> None:
        paras_list = get_class_init_args(type(self))
        paras_dict = {}
        for p in paras_list:
            if p in ['buffer_ds', 'num_workers']:
                continue
            paras_dict[p] = eval('self.' + p)
            # if type(paras_dict[p]) is list:
            #     paras_dict[p] = str(paras_dict[p])
        self.hparams.update(paras_dict)

    def read_data(self, dataset_dir: str = None, reader=None, formatters: dict = None):
        reader = self.default_reader if reader is None else reader
        if type(reader) is str:
            reader = eval('{0}.{0}'.format(reader))(dataset_dir=dataset_dir, reader_logger=self.model_logger)
        self.reader = reader
        if formatters is None:
            formatters = self.read_formatters()
        reader.read_train(filename=TRAIN_FILE, formatters=formatters)
        reader.read_validation(filename=VAL_FILE, formatters=formatters)
        reader.read_test(filename=TEST_FILE, formatters=formatters)
        return reader

    def read_formatters(self, formatters: dict = None) -> dict:
        current = {
            '^' + LABEL + '$': None,
            INT_F + '$': None,
            FLOAT_F + '$': None,
        }
        if formatters is None:
            return current
        return {**current, **formatters}

    def get_dataset(self, phase):
        if self.reader is None:
            self.model_logger.error('Model.Reader is None, Please read data first. (model.read_data(...))')
            return
        if phase == TRAIN_PHASE:
            if self.train_dataset is not None:
                return self.train_dataset
            self.train_dataset = eval('{0}.{0}'.format(self.default_dataset))(
                data=self.reader.train_data, reader=self.reader,
                model=self, buffer_ds=self.buffer_ds, phase=TRAIN_PHASE)
            return self.train_dataset
        if phase == VAL_PHASE:
            if self.val_dataset is not None:
                return self.val_dataset
            self.val_dataset = eval('{0}.{0}'.format(self.default_dataset))(
                data=self.reader.val_data, reader=self.reader, model=self, buffer_ds=self.buffer_ds, phase=VAL_PHASE)
            return self.val_dataset
        if phase == TEST_PHASE:
            if self.test_dataset is not None:
                return self.test_dataset
            self.test_dataset = eval('{0}.{0}'.format(self.default_dataset))(
                data=self.reader.test_data, reader=self.reader, model=self, buffer_ds=self.buffer_ds, phase=TEST_PHASE)
            return self.test_dataset
        if phase == PREDICT_PHASE:
            return eval('{0}.{0}'.format(self.default_dataset))(
                data=self.reader.test_data, reader=self.reader, model=self, buffer_ds=self.buffer_ds,
                phase=PREDICT_PHASE)
        self.model_logger.error("ERROR: unknown phase {}".format(phase))
        return

    def dataset_length(self, dataset):
        if type(dataset.data) is dict:
            for key in dataset.data:
                return len(dataset.data[key])
        return len(dataset.data)

    def dataset_get_item(self, dataset, index: int) -> dict:
        if dataset.buffer_ds > 0: return dataset.index_buffer[index]
        index_dict = {}
        for c in dataset.data:
            index_dict[c] = dataset.data[c][index]
        return index_dict

    def dataset_collate_batch(self, dataset, batch: list) -> dict:
        result_dict = {}
        for c in batch[0]:
            result_dict[c] = dataset.collate_stack([b[c] for b in batch])
        result_dict[PHASE] = dataset.phase
        return result_dict

    def dataset_get_dataloader(self, dataset: 'Dataset') -> torch.utils.data.DataLoader:
        if dataset.phase == TRAIN_PHASE:
            return torch.utils.data.DataLoader(
                dataset, batch_size=self.batch_size, shuffle=True, num_workers=self.num_workers,
                collate_fn=dataset.collate_batch, pin_memory=True)
        if dataset.phase == VAL_PHASE or dataset.phase == TEST_PHASE or dataset.phase == PREDICT_PHASE:
            return torch.utils.data.DataLoader(
                dataset, batch_size=self.eval_batch_size, shuffle=False, num_workers=self.num_workers,
                collate_fn=dataset.collate_batch, pin_memory=True)
        self.model_logger.error("ERROR: unknown phase {}".format(dataset.phase))
        return None

    def init_metrics(self, train_metrics=None, val_metrics=None, test_metrics=None, *args, **kwargs):
        if train_metrics is not None:
            if not isinstance(train_metrics, MetricsList):
                train_metrics = MetricsList(train_metrics)
            self.train_metrics = train_metrics
        if val_metrics is not None:
            if not isinstance(val_metrics, MetricsList):
                val_metrics = MetricsList(val_metrics)
            self.val_metrics = val_metrics
        if test_metrics is not None:
            if not isinstance(test_metrics, MetricsList):
                test_metrics = MetricsList(test_metrics)
            self.test_metrics = test_metrics

    def configure_optimizers(self):
        weight_p, bias_p = [], []
        for name, p in self.named_parameters():
            if not p.requires_grad:
                continue
            if 'bias' in name:
                bias_p.append(p)
            else:
                weight_p.append(p)
        if self.l2_bias == 1:
            optimize_dict = [{'params': weight_p + bias_p, 'weight_decay': self.l2}]
        else:
            optimize_dict = [{'params': weight_p, 'weight_decay': self.l2},
                             {'params': bias_p, 'weight_decay': 0.0}]

        optimizer_name = self.optimizer_name.lower()
        if optimizer_name == 'gd':
            self.model_logger.info("Optimizer: GD")
            optimizer = torch.optim.SGD(optimize_dict, lr=self.lr)
        elif optimizer_name == 'adagrad':
            self.model_logger.info("Optimizer: Adagrad")
            optimizer = torch.optim.Adagrad(optimize_dict, lr=self.lr)
        elif optimizer_name == 'adam':
            self.model_logger.info("Optimizer: Adam")
            optimizer = torch.optim.Adam(optimize_dict, lr=self.lr)
        else:
            self.model_logger.error("Unknown Optimizer: " + self.optimizer_name)
            assert self.optimizer_name in ['GD', 'Adagrad', 'Adam']
            optimizer = torch.optim.SGD(optimize_dict, lr=self.lr)
        return optimizer

    def init_weights(self) -> None:
        for n, p in self.named_parameters():
            if p.requires_grad:
                torch.nn.init.normal_(p, mean=0, std=0.01)

    def count_variables(self):
        """
        模型所有参数数目
        :return:
        """
        total_parameters = sum(p.numel() for p in self.parameters() if p.requires_grad)
        return total_parameters

    def init_logger(self, save_dir=MODEL_DIR, name=None, version=None):
        if name is None:
            name = os.path.basename(self.reader.dataset_dir)
        if version is None:
            model_name = self.__class__.__name__
            random_seed = os.environ["PL_GLOBAL_SEED"]
            self.save_hyperparameters()
            hash_code = hash_hparams(self.hparams)
            version = os.path.join(model_name, '_'.join([hash_code, random_seed]))
        self.log_dir = os.path.join(os.path.join(save_dir, name, version, ''))
        check_mkdir(self.log_dir)
        model_logger = logging.getLogger(self.__class__.__name__)
        while len(model_logger.handlers) > 0:
            model_logger.handlers.pop()
        logger_add_file_handler(model_logger, os.path.join(self.log_dir, LOG_F))
        model_logger.addHandler(logging.StreamHandler(sys.stdout))
        model_logger.setLevel(DEFAULT_LOGGER.level)
        self.model_logger = model_logger
        return save_dir, name, version

    def init_trainer(self, save_dir=MODEL_DIR, name=None, version=None, **kwargs) -> pl.Trainer:
        save_dir, name, version = self.init_logger(save_dir=save_dir, name=name, version=version)

        default_para = copy.deepcopy(DEFAULT_TRAINER_ARGS)
        default_para.update(kwargs)
        default_para['callbacks'].append(EarlyStopping(mode='max', patience=self.es_patience, verbose=True))
        default_para['callbacks'].append(
            ModelCheckpoint(mode='max', monitor=EARLY_STOP_ON, save_last=True,
                            dirpath=os.path.join(self.log_dir, CKPT_DIR), filename=CKPT_F,
                            verbose=self.model_logger.level <= logging.DEBUG))
        # if GLOBAL_ARGS['pbar']:
        #     default_para['callbacks'].append(LitProgressBar())
        # else:
        #     default_para['progress_bar_refresh_rate'] = 0
        default_para['callbacks'].append(LitProgressBar(refresh_rate=default_para['progress_bar_refresh_rate']))
        csv_logger = CSVLogger(save_dir=save_dir, name=name, version=version)
        # tb_logger = TensorBoardLogger(save_dir=save_dir, description=version, name=dataset_name)
        default_para['logger'].append(csv_logger)
        # default_para['logger'].append(tb_logger)
        self.trainer = pl.Trainer.from_argparse_args(Namespace(**default_para))
        return self.trainer

    def fit(self, train_data=None, val_data=None, trainer=None, **kwargs):
        if trainer is None:
            trainer = self.trainer if self.trainer is not None else self.init_trainer(**kwargs)
        if train_data is None:
            train_data = self.get_dataset(phase=TRAIN_PHASE)
        if val_data is None:
            val_data = self.get_dataset(phase=VAL_PHASE)
        if isinstance(train_data, Dataset.Dataset):
            train_data = self.dataset_get_dataloader(train_data)
        if isinstance(val_data, Dataset.Dataset):
            val_data = self.dataset_get_dataloader(val_data)
        return trainer.fit(model=self, train_dataloader=train_data, val_dataloaders=val_data)

    def test(self, test_data=None, trainer=None, **kwargs):
        if trainer is None:
            trainer = self.trainer if self.trainer is not None else self.init_trainer(**kwargs)
        if test_data is None:
            test_data = self.get_dataset(phase=TEST_PHASE)
        if isinstance(test_data, Dataset.Dataset):
            test_data = self.dataset_get_dataloader(test_data)
        return trainer.test(model=self, test_dataloaders=test_data)

    def predict(self, predict_data=None, trainer=None, **kwargs):
        if trainer is None:
            trainer = self.trainer if self.trainer is not None else self.init_trainer(**kwargs)
        if predict_data is None:
            predict_data = self.get_dataset(phase=PREDICT_PHASE)
        if isinstance(predict_data, Dataset.Dataset):
            predict_data = self.dataset_get_dataloader(predict_data)
        return trainer.predict(model=self, dataloaders=predict_data)

    def init_modules(self, *args, **kwargs) -> None:
        self.init_weights()
        return

    def forward(self, batch, *args, **kwargs):
        return batch

    def loss_func(self, batch, out_dict, *args, **kwargs):
        return

    def training_step(self, batch, batch_idx, *args, **kwargs):
        out_dict = self.forward(batch)
        loss = self.loss_func(batch, out_dict)
        self.log('{}_loss'.format(self.loss_type), loss, on_step=True)
        if self.train_metrics is not None and len(self.train_metrics.metrics) > 0:
            self.train_metrics.update(out_dict)
        out_dict[LOSS] = loss
        return out_dict

    def validation_step(self, batch, *args, **kwargs):
        out_dict = self.forward(batch)
        if self.val_metrics is not None and len(self.val_metrics.metrics) > 0:
            self.val_metrics.update(out_dict)
        else:
            out_dict[LOSS] = self.loss_func(batch, out_dict)
        return out_dict

    def test_step(self, batch, *args, **kwargs):
        out_dict = self.forward(batch)
        if self.test_metrics is not None and len(self.test_metrics.metrics) > 0:
            self.test_metrics.update(out_dict)
        else:
            out_dict[LOSS] = self.loss_func(batch, out_dict)
        return out_dict

    def on_save_checkpoint(self, checkpoint) -> None:
        self.save_hyperparameters()
        self.logger.log_hyperparams(self.hparams)

    def on_load_checkpoint(self, checkpoint) -> None:
        self.save_hyperparameters()
        self.init_modules()

    def load_model(self, checkpoint_path=None, hparams_file=None):
        checkpoint = {}
        if checkpoint_path is not None:
            if os.path.exists(checkpoint_path):
                self.model_logger.info('load model from {}'.format(checkpoint_path))
                checkpoint = torch.load(checkpoint_path)
            else:
                self.model_logger.warning('checkpoint_path does not exist: {}'.format(checkpoint_path))
        hparams = {}
        if self.CHECKPOINT_HYPER_PARAMS_KEY in checkpoint:
            hparams = checkpoint[self.CHECKPOINT_HYPER_PARAMS_KEY]
        if hparams_file is not None:
            if os.path.exists(hparams_file):
                self.model_logger.info('load hparams from {}'.format(hparams_file))
                hparams.update(yaml.load(open(hparams_file, 'r'), Loader=yaml.FullLoader))
            else:
                self.model_logger.warning('hparams_file does not exist: {}'.format(hparams_file))
        self.hparams.update(hparams)
        for p in self.hparams:
            exec("self.{} = self.hparams['{}']".format(p, p))
        if STATE_DICT in checkpoint:
            try:
                self.load_state_dict(checkpoint[STATE_DICT])
            except:
                self.init_modules()
                self.load_state_dict(checkpoint[STATE_DICT])

    def on_train_end(self) -> None:
        if self.log_dir is not None:
            checkpoint_path = os.path.join(self.log_dir, CKPT_DIR, CKPT_F + '.ckpt')
            hparams_file = os.path.join(self.log_dir, 'hparams.yaml')
            self.load_model(checkpoint_path=checkpoint_path, hparams_file=hparams_file)

    def training_epoch_end(self, outputs) -> None:
        if self.train_metrics is not None and len(self.train_metrics.metrics) > 0:
            metrics = self.train_metrics.compute(reset=True)
        else:
            loss = torch.stack([o[LOSS] for o in outputs]).sum()
            metrics = {LOSS: loss}
        self.train_metrics_buf.append(metrics)
        train_metrics = {}
        for key in metrics:
            train_metrics['train_' + key] = metrics[key]
        self.log_dict(train_metrics)

    def on_train_epoch_end(self, outputs) -> None:
        if len(self.train_metrics_buf) > 0:
            prefix = '[Train Epoch {:4}]'.format(self.current_epoch + 1)
            self.model_logger.debug(os.linesep + prefix + format_log_metrics_list(self.train_metrics_buf))

    def validation_epoch_end(self, outputs):
        if self.val_metrics is not None and len(self.val_metrics.metrics) > 0:
            metrics = self.val_metrics.compute(reset=True)
            es_name = self.val_metrics.metrics_str[0]
            early_stop_on = metrics[es_name]
            if es_name in METRICS_SMALLER:
                early_stop_on = -early_stop_on
        else:
            loss = torch.stack([o[LOSS] for o in outputs]).sum()
            early_stop_on = -loss
            metrics = {LOSS: loss}
        self.val_metrics_buf.append(metrics)
        self.log(EARLY_STOP_ON, early_stop_on)
        val_metrics = {}
        for key in self.val_metrics.metrics_str:
            val_metrics['val_' + key] = metrics[key]
        self.log_dict(val_metrics)

    def on_validation_epoch_end(self) -> None:
        if len(self.val_metrics_buf) > 0:
            if 0 < self.trainer.val_check_interval < 1:
                frequency = int(1.0 / self.trainer.val_check_interval)
                mod = (len(self.val_metrics_buf) - 1) % frequency
                epoch = int((len(self.val_metrics_buf) - 1) / frequency)
                if epoch > 0 and mod == 0:
                    epoch += self.trainer.val_check_interval * frequency - 1
                else:
                    epoch += mod * self.trainer.val_check_interval
                prefix = '[Validation Epoch {:>6}]'.format('{:.2f}'.format(epoch))
            else:
                epoch = self.trainer.check_val_every_n_epoch * (len(self.val_metrics_buf) - 1)
                prefix = '[Validation Epoch {:4}]'.format(epoch)
            info_line = prefix + format_log_metrics_list(self.val_metrics_buf)
            if WINDOW_WIDTH > len(info_line):
                info_line = info_line + ' ' * (WINDOW_WIDTH - len(info_line))
            self.model_logger.info(os.linesep + info_line)

    def test_epoch_end(self, outputs):
        if self.test_metrics is not None and len(self.test_metrics.metrics) > 0:
            metrics = self.test_metrics.compute(reset=True)
        else:
            loss = torch.stack([o[LOSS] for o in outputs]).sum()
            metrics = {LOSS: loss}
        self.test_metrics_buf.append(metrics)
        test_metrics = {}
        for key in self.test_metrics.metrics_str:
            test_metrics['test_' + key] = metrics[key]
        self.log_dict(test_metrics, logger=False)

    def on_test_epoch_end(self) -> None:
        if len(self.test_metrics_buf) > 0:
            prefix = '[Test {:4}]'.format(len(self.test_metrics_buf))
            info_line = prefix + format_log_metrics_list(self.test_metrics_buf)
            if WINDOW_WIDTH > len(info_line):
                info_line = info_line + ' ' * (WINDOW_WIDTH - len(info_line))
            self.model_logger.info(os.linesep + info_line)

    def get_progress_bar_dict(self):
        # don't show the version number
        items = super().get_progress_bar_dict()
        version = '' if self.log_dir is None else self.log_dir
        items['v_num'] = version.strip('/').split('/')[-1][:5]
        return items
