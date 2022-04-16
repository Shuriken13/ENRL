# coding=utf-8
import argparse
from argparse import ArgumentParser
from ..models import *
from ..configs.settings import *
from ..tasks import *


def add_trainer_args(parent_parser):
    parser = ArgumentParser(parents=[parent_parser], add_help=False)
    parser.add_argument('--check_val_every_n_epoch', type=int, default=DEFAULT_TRAINER_ARGS['check_val_every_n_epoch'],
                        help='Check val every n train epochs.')
    parser.add_argument('--fast_dev_run', type=int, default=DEFAULT_TRAINER_ARGS['fast_dev_run'],
                        help='Runs n if set to n (int) else 1 if set to True batch(es) of train, val and test to find any bugs.')
    parser.add_argument('--gradient_clip_val', type=float, default=DEFAULT_TRAINER_ARGS['gradient_clip_val'],
                        help='Gradient clipping value.')
    parser.add_argument('--max_epochs', type=int, default=DEFAULT_TRAINER_ARGS['max_epochs'],
                        help='Stop training once this number of epochs is reached.')
    parser.add_argument('--min_epochs', type=int, default=DEFAULT_TRAINER_ARGS['min_epochs'],
                        help='Force training for at least these many epochs.')
    parser.add_argument('--profiler', type=str, default=DEFAULT_TRAINER_ARGS['profiler'],
                        help='To profile individual steps during training and assist in identifying bottlenecks.')
    parser.add_argument('--val_check_interval', type=float, default=DEFAULT_TRAINER_ARGS['val_check_interval'],
                        help='How often within one training epoch to check the validation set. '
                             'Use (float) to check within a training epoch.'
                             'use (int) to check every n steps (batches)')
    return parser


def parse_cmd_args(task_name='Task'):
    # # task arguments
    task_name = eval('{0}.{0}'.format(task_name))
    task_parser = argparse.ArgumentParser(description='Task Args', add_help=False)
    task_parser = task_name.add_task_args(task_parser)
    task_args, _ = task_parser.parse_known_args()
    # print(task_args)

    # # model arguments
    model_name = eval('{0}.{0}'.format(task_args.model_name))
    model_parser = argparse.ArgumentParser(description='Model Args', add_help=False)
    model_parser = model_name.add_model_specific_args(model_parser)
    model_args, _ = model_parser.parse_known_args()
    # print(model_args)

    # # trainer arguments
    trainer_parser = argparse.ArgumentParser(description='Trainer Args', add_help=False)
    trainer_parser = add_trainer_args(trainer_parser)
    trainer_args, _ = trainer_parser.parse_known_args()
    # print(trainer_args)

    # # all arguments
    parser = argparse.ArgumentParser(parents=[task_parser, model_parser, trainer_parser],
                                     description=task_args.model_name)
    args = parser.parse_args()
    return args, task_args, model_args, trainer_args
