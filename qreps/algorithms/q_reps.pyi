from typing import Any

from rllib.algorithms.reps import REPS
from rllib.value_function import AbstractQFunction

class QREPS(REPS):
    q_function: AbstractQFunction
    def __init__(
        self, q_function: AbstractQFunction, *args: Any, **kwargs: Any
    ) -> None: ...
