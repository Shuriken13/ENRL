# coding=utf-8

import torch


def hard_binary(values):
    hard = values.ge(0.5).float()
    hard = (hard - values).detach() + values
    return hard


def hard_max(values):
    _, index = values.max(dim=-1)  # ?
    hard = torch.nn.functional.one_hot(index, num_classes=values.size(-1))  # ? * c
    hard = (hard - values).detach() + values  # ? * c
    return hard  # ? * c


def hard_softmax_func(logits, tau=1.0):
    probs = torch.nn.functional.softmax(logits / tau, dim=-1)  # ? * c
    return hard_max(probs)  # ? * c


class HardSoftmax(torch.nn.Module):
    def __init__(self, tau=1.0):
        super().__init__()
        self.tau = tau

    def forward(self, logits):
        return hard_softmax_func(logits=logits, tau=self.tau)


def gumbel_softmax_func(logits, training, random_r=0.5, eps=1e-20, tau=1.0, hard_r=1.0, hard=True):
    if not training:
        return hard_softmax_func(logits, tau=tau) if hard \
            else torch.nn.functional.softmax(logits / tau, dim=-1)  # ? * c

    u = torch.rand(logits.size(), device=logits.device)  # ? * c
    u = -torch.log(-torch.log(u + eps) + eps)  # ? * c
    r = torch.rand(logits.size()[:-1], device=logits.device).le(random_r).unsqueeze(dim=-1)  # ? * 1
    logits = logits + u * r  # ? * c

    if hard_r >= 1.0:
        return hard_softmax_func(logits, tau=tau)  # ? * c
    if hard_r <= 0.0:
        return torch.nn.functional.softmax(logits / tau, dim=-1)  # ? * c

    hard_result = hard_softmax_func(logits, tau=tau)  # ? * c
    soft_result = torch.nn.functional.softmax(logits / tau, dim=-1)  # ? * c
    r = torch.rand(logits.size()[:-1], device=logits.device).le(hard_r).unsqueeze(dim=-1).float()  # ? * 1
    return hard_result * r + soft_result * (-r + 1)


class GumbelSoftmax(torch.nn.Module):
    def __init__(self, random_r=0.5, eps=1e-20, tau=1.0, hard_r=1.0, hard=True):
        super().__init__()
        self.random_r = random_r
        self.eps = eps
        self.tau = tau
        self.hard = hard
        self.hard_r = hard_r

    def forward(self, logits):
        return gumbel_softmax_func(logits=logits, training=self.training, random_r=self.random_r,
                                   eps=self.eps, tau=self.tau, hard_r=self.hard_r, hard=self.hard)
