"""Python Script Template."""
import torch
from rllib.dataset.datatypes import Observation
from rllib.policy import AbstractPolicy, RandomPolicy, TabularPolicy

from qreps.util.utilities import compute_exact_td


class ExactREPSPolicy(AbstractPolicy):
    """Exact computation of REPS policy."""

    def __init__(
        self, value_function, num_actions, eta, transitions, rewards, gamma, prior=None
    ):
        super().__init__(
            dim_state=(),
            dim_action=(),
            num_states=value_function.num_states,
            num_actions=num_actions,
        )
        self.value_function = value_function
        self.transitions = transitions
        self.rewards = rewards
        self.gamma = gamma
        self.eta = eta
        self.counter = 0
        if prior is None:
            prior = RandomPolicy(
                dim_state=self.dim_state,
                dim_action=self.dim_state,
                num_states=self.num_states,
                num_actions=self.num_actions,
            )
        self.prior = prior

    def reset(self):
        """Reset policy and increase counter."""
        super().reset()
        self.counter += 1

    def forward(self, state):
        """Return policy logits."""
        td = compute_exact_td(
            value_function=self.value_function,
            observation=Observation(state=state),
            transitions=self.transitions,
            rewards=self.rewards,
            gamma=self.gamma,
            support="state",
        )
        return self.eta * td * self.counter

    def tabular_representation(self):
        """Get a tabular representation of the policy."""
        policy = TabularPolicy(self.num_states, self.num_actions)
        for state in range(self.num_states):
            state = torch.tensor(state)
            policy.set_value(state, self(state).clone())
        return policy
