"""Python Script Template."""

import torch.nn as nn
from rllib.util.neural_networks.utilities import (deep_copy_module,
                                                  freeze_parameters)

from qreps.util.utilities import accumulate_parameters


class AccumulativeModule(nn.Module):
    """Function with accumulation."""

    def __init__(self, func):
        super().__init__()
        # self.__dict__.update(**func.__dict__)
        self.func = func
        running_func = deep_copy_module(func)
        freeze_parameters(running_func)
        self.running_func = running_func
        self.count = 0
        for key, value in func.__dict__.items():
            if key not in self.__dict__.keys():
                self.__dict__[key] = value

    def reset(self):
        """Update the value function."""
        accumulate_parameters(self.running_func, self.func, self.count)
        freeze_parameters(self.running_func)
        self.count += 1

    def forward(self, *args, **kwargs):
        """Combine mean with current function."""
        prior = self.count * self.running_func(*args, **kwargs)
        this = self.func(*args, **kwargs)
        return (prior + this) / (self.count + 1)
