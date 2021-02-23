"""Python Script Template."""
from itertools import chain

import torch
from torch.optim import SGD

from qreps.policy.q_reps_policy import QREPSPolicy
from qreps.util.accumulative_module import AccumulativeModule
from qreps.util.utilities import (
    compute_exact_td,
    get_default_q_function,
    get_primal_q_reps,
)
from qreps.value_function.soft_value_function import SoftValueFunction

from .reps_exact import REPSExactAgent, REPSExactAlgorithm


class QREPSExactAlgorithm(REPSExactAlgorithm):
    """Implementaiton of Q-REPS algorithm.

    It uses a model to compute the dual.
    It extracts the policy from the q-function.
    """

    def __init__(self, q_function, alpha=None, *args, **kwargs):
        q_function = AccumulativeModule(q_function)
        self.alpha = alpha
        super().__init__(*args, **kwargs)
        policy = QREPSPolicy(q_function, self.alpha)
        self.q_function = q_function
        self.set_policy(policy)
        if self.alpha is not None:
            self.critic = SoftValueFunction(q_function, param=self.alpha)
            self.critic_target = self.critic

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
        td = compute_exact_td(
            value_function=self.critic,
            observation=observation,
            transitions=self.transitions,
            rewards=self.rewards,
            gamma=self.gamma,
            support=self.support,
        )
        td = td.squeeze(-1) + v - q
        if self.alpha is not None:
            dual = self.y0[idx] * torch.exp(self.eta * td)
        else:
            adv = q - v
            dual = self.y0[idx] * (torch.exp(self.eta * td) + torch.exp(self.eta * adv))
        return torch.log(dual.sum()) / self.eta

    def reset(self):
        """Reset episode."""
        super().reset()
        self.q_function.reset()


class QREPSExactAgent(REPSExactAgent):
    """Exact Q REPS agent."""

    def __init__(self, q_function, lr, optimizer_=SGD, alpha=None, *args, **kwargs):
        super().__init__(lr=lr, *args, **kwargs)
        self.algorithm = QREPSExactAlgorithm(
            q_function=q_function, alpha=alpha, *args, **kwargs
        )
        self.policy = self.algorithm.policy
        self.optimizer = optimizer_(
            chain(
                self.algorithm.q_function.parameters(),
                self.algorithm.critic.parameters(),
            ),
            lr=lr,
        )

    @classmethod
    def default(
        cls,
        environment,
        eta=1.0,
        alpha=None,
        function_approximation="tabular",
        *args,
        **kwargs,
    ):
        """Get default agent."""
        return super().default(
            environment,
            q_function=get_default_q_function(environment, function_approximation),
            eta=eta,
            alpha=alpha,
            function_approximation=function_approximation,
            *args,
            **kwargs,
        )
