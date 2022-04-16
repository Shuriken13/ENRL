# coding=utf-8

import torch
import torchmetrics

from ..configs.constants import *
from ..metrics import metrics as mm


class MetricsList(torch.nn.Module):
    support_metrics = {
        'accuracy': mm.Accuracy,
        'auc': mm.AUROC,
        'f1': mm.F1,
        'mae': mm.MAE,
        'mse': mm.MSE,
        'precision': mm.Precision,
        'recall': mm.Recall,
        'rmse': mm.RMSE,
    }

    def __init__(self, metrics, **kwargs):
        super().__init__()
        self.metrics_kwargs = kwargs
        self.metrics_str = self.parse_metrics_str(metrics)
        self.metrics = torch.nn.ModuleDict()
        self.init_metrics()

    def parse_metrics_str(self, metrics_str: str):
        if type(metrics_str) is str:
            metrics_str = metrics_str.lower().strip().split(METRIC_SPLITTER)
        metrics = []
        for metric in metrics_str:
            metric = metric.strip()
            if metric == '' or metric not in self.support_metrics:
                continue
            metrics.append(metric)
        return metrics

    def init_metrics(self):
        for metric in self.metrics_str:
            metric = metric.strip()
            if metric in self.metrics:
                continue
            self.metrics[metric] = self.support_metrics[metric](**self.metrics_kwargs)

    def forward(self, *args, **kwargs):
        self.update(*args, **kwargs)
        if self.compute_on_step:
            return self.compute()

    def update(self, output: dict) -> None:
        for key in self.metrics:
            metric = self.metrics[key]
            metric.update(output)

    def compute(self, reset=False):
        result = {}
        for metric in self.metrics:
            metric_result = self.metrics[metric].compute()
            for k in metric_result:
                result[k] = metric_result[k]
        if reset:
            self.reset()
        return {k: result[k] for k in self.metrics_str}

    def reset(self):
        for metric in self.metrics:
            self.metrics[metric].reset()
