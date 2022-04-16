# coding=utf-8


import torchmetrics
import torch

from ..configs.constants import *

METRICS_SMALLER = {
    'rmse', 'mse', 'mae', 'loss'
}


class Accuracy(torchmetrics.Accuracy):
    def update(self, output: dict):
        return super().update(preds=output[PREDICTION], target=output[LABEL])

    def compute(self):
        return {'accuracy': super().compute()}


class AUROC(torchmetrics.AUROC):
    def __init__(self, pos_label: int = None, *args, **kwargs):
        super().__init__(pos_label=1, *args, **kwargs)

    def update(self, output: dict):
        return super().update(preds=output[PREDICTION], target=output[LABEL])

    def compute(self):
        try:
            return {'auc': super().compute()}
        except Exception as e:
            if str(e).startswith('No positive samples in targets') or \
                    str(e).startswith('No negative samples in targets'):
                return {'auc': torch.tensor(0.0, device=self.preds[0].device)}
            else:
                raise e


class F1(torchmetrics.F1):
    def update(self, output: dict):
        return super().update(preds=output[PREDICTION], target=output[LABEL])

    def compute(self):
        return {'f1': super().compute()}


class MAE(torchmetrics.MeanAbsoluteError):
    def update(self, output: dict):
        return super().update(preds=output[PREDICTION], target=output[LABEL])

    def compute(self):
        return {'mae': super().compute()}


class MSE(torchmetrics.MeanSquaredError):
    def update(self, output: dict):
        return super().update(preds=output[PREDICTION], target=output[LABEL])

    def compute(self):
        return {'mse': super().compute()}


class Precision(torchmetrics.Precision):
    def update(self, output: dict):
        return super().update(preds=output[PREDICTION], target=output[LABEL])

    def compute(self):
        return {'precision': super().compute()}


class Recall(torchmetrics.Recall):
    def update(self, output: dict):
        return super().update(preds=output[PREDICTION], target=output[LABEL])

    def compute(self):
        return {'recall': super().compute()}


class RMSE(torchmetrics.MeanSquaredError):
    def update(self, output: dict):
        return super().update(preds=output[PREDICTION], target=output[LABEL])

    def compute(self):
        return {'rmse': super().compute().sqrt()}
