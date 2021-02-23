"""Python Script Template."""
from itertools import chain

import torch
from torch.optim import SGD

from qreps.policy.q_reps_policy import QREPSPolicy
from qreps.util.accumulative_module import AccumulativeModule
from qreps.util.utilities import get_primal_q_reps, sample_td
from qreps.value_function.soft_value_function import SoftValueFunction

from .q_reps_exact import QREPSExactAgent
from .reps_saddle_exact_policy import REPSSaddleAlgorithm


class QREPSSaddleAlgorithm(REPSSaddleAlgorithm):
    """Implementation of SaddleQREPSAlgorithm.

    It uses a q-function to compute the policy.
    It uses the saddle-point formulation to get unbiased samples of the dual.
    """

    def __init__(self, critic, q_function, alpha=None, *args, **kwargs):
        q_function = AccumulativeModule(q_function)
        self.alpha = alpha
        if self.alpha is not None:
            critic = SoftValueFunction(q_function, param=alpha)
        else:
            alpha = self.eta

        super().__init__(critic=critic, *args, **kwargs)
        self.q_function = q_function
        policy = QREPSPolicy(q_function, alpha)
        self.set_policy(policy)
        self.y_adv = torch.zeros(1)
        self.sum_g_adv = torch.zeros(1)

    @property
    def constraint_violation(self):
        """Get un constraint violation."""
        mu, d = get_primal_q_reps(
            prior=self.prior,
            eta=self.eta,
            value_function=self.critic,
            q_function=self.q_function,
            transitions=self.transitions,
            rewards=self.rewards,
            gamma=self.gamma,
        )
        next_nu = torch.einsum("ijk,ij->k", self.transitions, mu)

        flow_constraint = (d.sum(-1) - next_nu).abs().sum()
        equality_constraint = (mu - d).abs().sum()
        self._info.update(
            flow_constraint=flow_constraint, equality_constraint=equality_constraint
        )
        return flow_constraint + equality_constraint

    def init(self, observation):
        """Initialize Saddle point variables."""
        super().init(observation)
        self.y_adv = self.y0.clone()
        self.sum_g_adv = (
            torch.distributions.Categorical(probs=self.y_adv).logits / self.saddle_lr
        )

    def update_multipliers(self, idx):
        """Update dual variables."""
        td, adv = self._info["td"], self._info["advantage"]
        self.y_adv = self._update_multipliers(self.y_adv, self.sum_g_adv, adv, idx)
        self.y_td = self._update_multipliers(self.y_td, self.sum_g_td, td, idx)

    def dual(self, observation, idx):
        """Compute dual and td."""
        state, action = observation.state, observation.action
        if state.ndim > 1:
            state = state.squeeze(1)
        if action.ndim > 1:
            action = action.squeeze(1)

        if self.support == "state":
            q, v = self.q_function(state), self.critic(state).unsqueeze(-1)
        elif self.support == "state-action":
            q, v = self.q_function(state, action), self.critic(state)
        else:
            raise NotImplementedError(f"{self.support} not implemented.")
        td = sample_td(
            value_function=self.critic,
            observation=observation,
            simulator=self.simulator,
            gamma=self.gamma,
            num_samples=self.num_samples,
            support=self.support,
            num_actions=self.num_actions,
        )

        td = td + v - q
        adv = q - v
        if self.alpha is not None:
            dual = self.y_td[idx].detach() * td
        else:
            dual = self.y_td[idx].detach() * td + self.y_adv[idx].detach() * adv

        self._info.update(td=td, advantage=adv)
        return dual.sum()

    def reset(self):
        """Reset episode."""
        super().reset()
        self.q_function.reset()


class QREPSSaddleAgent(QREPSExactAgent):
    """Saddle REPS agent."""

    def __init__(self, lr, optimizer_=SGD, *args, **kwargs):
        super().__init__(lr=lr, *args, **kwargs)
        self.algorithm = QREPSSaddleAlgorithm(*args, **kwargs)
        self.policy = self.algorithm.policy
        self.optimizer = optimizer_(
            chain(
                self.algorithm.q_function.parameters(),
                self.algorithm.critic.parameters(),
            ),
            lr=lr,
        )
