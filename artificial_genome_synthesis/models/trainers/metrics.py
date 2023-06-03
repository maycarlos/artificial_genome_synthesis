from typing import Callable, Union

import torch
from ignite.metrics import Metric


class Average(Metric):
    def __init__(
        self,
        loss_fn: Callable,
        output_transform: Callable = lambda x: x,
        batch_size: Callable = len,
        device: Union[str, torch.device] = torch.device("cpu"),
    ):
        super(Average, self).__init__(output_transform, device=device)
        self._loss_fn = loss_fn
        self._batch_size = batch_size
        self.n = 0

    def reset(self) -> None:
        self._sum = torch.tensor(0.0).cuda()
        self.n = 0

    def update(self, x):
        loss = self._loss_fn(x[0], x[1])
        self._sum += loss
        self.n += 1

    def compute(self) -> float:
        return self._sum.item() / self.n
