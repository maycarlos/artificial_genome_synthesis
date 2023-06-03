# TODO CVAE
from ...base_classes import BaseModel
import torch.nn


class ConditionalVariationalAutoEncoder(BaseModel):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)

    def forward(self,X):
        return super().forward()
