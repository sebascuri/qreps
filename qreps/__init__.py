import gym
import torch

import qreps.environment

torch.set_default_dtype(torch.float32)
gym.logger.set_level(gym.logger.ERROR)
