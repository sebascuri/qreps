"""Python Script Template."""

import torch
from torch.optim import SGD

from qreps.util.utilities import compute_expected_etd, sample_td

from .reps_exact import REPSExactAgent, REPSExactAlgorithm


class REPSSampleExactAlgorithm(REPSExactAlgorithm):
    """Implementation of Sample-based REPS algorithm.

    It uses num_samples to compute a biased estimate of the dual.
    If num_samples = 0 it uses a model to compute the exact biased estimate.

    It uses the model to compute the policy.
    """

    def __init__(self, simulator, num_samples=1, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.simulator = simulator
        self.num_samples = num_samples

    @property
    def constraint_violation(self):
        """Get un constraint violation.

        As the optimization is biased, the result is not stationary.
        """
        return torch.tensor(0.0)

    def dual(self, observation, idx):
        """Compute dual and td."""
        if self.num_samples <= 0:  # BATCH REPS
            expected_etd = compute_expected_etd(
                value_function=self.critic,
                observation=observation,
                transitions=self.simulator.transition_kernel,
                rewards=self.simulator.rewards,
                gamma=self.gamma,
                eta=self.eta,
                support=self.support,
            )
            dual = self.y0[idx] * expected_etd

        else:
            td = sample_td(
                value_function=self.critic,
                observation=observation,
                simulator=self.simulator,
                gamma=self.gamma,
                num_samples=self.num_samples,
                support=self.support,
                num_actions=self.num_actions,
            )
            dual = self.y0[idx] * torch.exp(self.eta * td)
        return torch.log(dual.sum()) / self.eta


class REPSSampleExactAgent(REPSExactAgent):
    """Sample REPS agent."""

    def __init__(
        self,
        num_samples,
        lr,
        simulator,
        gamma,
        optimizer_=SGD,
        *args,
        **kwargs,
    ):
        super().__init__(lr=lr, gamma=gamma, *args, **kwargs)
        self.algorithm = REPSSampleExactAlgorithm(
            num_samples=num_samples,
            simulator=simulator,
            gamma=self.gamma,
            *args,
            **kwargs,
        )
        self.policy = self.algorithm.policy
        self.optimizer = optimizer_(self.algorithm.critic.parameters(), lr=lr)
