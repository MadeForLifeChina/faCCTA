import abc
import os
import warnings
from warnings import warn

import numpy as np
import torch

from batchgenerators.transforms.abstract_transforms import Compose


class CustomTransform(Compose):
    def __init__(self, transforms):
        super(CustomTransform, self).__init__(transforms)
        print(f"==> Using custom transform from {__file__}")

        # # Use Nearest augmentation
        # for t_i in range(len(self.transforms)):
        #     t = self.transforms[t_i]
        #     if t.__class__.__name__ == 'SpatialTransform':
        #         self.transforms[t_i].order_data = 0

    def __call__(self, **data_dict):
        for t in self.transforms:
            data_dict = t(**data_dict)

        return data_dict
    def __repr__(self):
        return str(type(self).__name__) + " ( " + repr(self.transforms) + " )"


class CustomTransformVal(CustomTransform):
    def __init__(self, transforms):
        super(CustomTransformVal, self).__init__(transforms)
        print(f"==> Using custom transform from {__file__}")
