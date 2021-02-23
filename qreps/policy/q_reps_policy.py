"""Python Script Template."""
import torch
from rllib.policy import AbstractQFunctionPolicy, TabularPolicy


class QREPSPolicy(AbstractQFunctionPolicy):
    """Implementation of a softmax policy with some small off-set for stability."""

    def __init__(self, q_function, param, *args, **kwargs):
        super().__init__(q_function, param)
        self.counter = 0

    @property
    def temperature(self):
        """Return temperature."""
        return self.param()

    def forward(self, state):
        """See `AbstractQFunctionPolicy.forward'."""
        q_val = self.q_function(state)
        return self.temperature * q_val * self.counter

    def reset(self):
        """Reset parameters and update counter."""
        self.counter += 1

    def tabular_representation(self):
        """Get a tabular representation of the policy."""
        policy = TabularPolicy(self.num_states, self.num_actions)
        for state in range(self.num_states):
            state = torch.tensor(state)
            policy.set_value(state, self(state).clone())
        return policy
