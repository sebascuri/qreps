"""Python Script Template."""
import torch
from rllib.util.parameter_decay import Constant, ParameterDecay
from rllib.value_function import AbstractValueFunction, NNQFunction


class SoftValueFunction(AbstractValueFunction):
    """Soft value function.

    Parameters
    ----------
    q_function: AbstractQFunction
        q _function.
    policy: AbstractPolicy
        q _function.
    num_samples: int, optional (default=15).
        Number of states in discrete environments.
    """

    def __init__(self, q_function, param, *args, **kwargs):
        if not q_function.discrete_action:
            raise NotImplementedError
        super().__init__(
            dim_state=q_function.dim_state,
            num_states=q_function.num_states,
        )
        self.q_function = q_function
        if not isinstance(param, ParameterDecay):
            param = Constant(param)
        self.param = param

    @property
    def alpha(self):
        """Return temperature."""
        return self.param()

    def update(self):
        """Update policy parameters."""
        self.param.update()

    @classmethod
    def default(cls, environment, *args, **kwargs):
        """See AbstractValueFunction.default."""
        q_function = NNQFunction.default(environment, *args, **kwargs)
        return super().default(environment, q_function=q_function, param=1)

    def forward(self, state):
        """Get value of the value-function at a given state."""
        q_value = self.q_function(state) * self.alpha
        return torch.logsumexp(q_value, dim=-1) / self.alpha
