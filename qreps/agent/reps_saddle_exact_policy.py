"""Python Script Template."""
import torch
from torch.optim import SGD

from qreps.util.utilities import get_primal_reps, sample_td

from .reps_exact import REPSExactAgent
from .reps_sample_exact_policy import REPSSampleExactAlgorithm


class REPSSaddleAlgorithm(REPSSampleExactAlgorithm):
    """Implementation of SaddleREPSAlgorithm.

    It uses a model to compute the policy.
    It uses the saddle-point formulation to get unbiased samples of the dual.
    """

    def __init__(self, saddle_lr, saddle_mixing, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.saddle_lr = saddle_lr
        self.saddle_mixing = saddle_mixing
        self.y_td = torch.zeros(1)
        self.sum_g_td = torch.zeros(1)

    @property
    def constraint_violation(self):
        """Get un constraint violation."""
        mu = get_primal_reps(
            prior=self.prior,
            eta=self.eta,
            value_function=self.critic,
            transitions=self.transitions,
            rewards=self.rewards,
            gamma=self.gamma,
        )
        nu = mu.sum(-1)
        next_nu = torch.einsum("ijk,ij->k", self.transitions, mu)
        return (nu - next_nu).abs().sum()

    def init(self, observation):
        """Initialize Saddle point variables."""
        super().init(observation)
        self.y_td = self.y0.clone()
        self.sum_g_td = (
            torch.distributions.Categorical(probs=self.y_td).logits / self.saddle_lr
        )

    def _update_multipliers(self, y, sum_g, error, idx):
        with torch.no_grad():
            _, inverse_indexes, counts = idx.unique(
                return_inverse=True, return_counts=True
            )

            # unique_indexes[inverse_indexes] == idx
            importance_weight = y[idx] * len(idx)
            g = error - 1 / self.eta * (torch.log(y[idx] / self.y0[idx]))
            sum_g[idx] += g * importance_weight * counts[inverse_indexes]
            y_dist = torch.distributions.Categorical(
                logits=(self.saddle_lr * sum_g).reshape(-1)
            )
            if self.support == "state":
                w = y_dist.probs.reshape(-1, self.num_actions)
            elif self.support == "state-action":
                w = y_dist.probs
            else:
                raise NotImplementedError(f"{self.support} not implemented.")
            return (1 - self.saddle_mixing) * w + self.saddle_mixing / w.shape[0]

    def update_multipliers(self, idx):
        """Update dual variables."""
        td = self._info["td"]
        self.y_td = self._update_multipliers(self.y_td, self.sum_g_td, td, idx)

    def dual(self, observation, idx):
        """Compute dual and td."""
        td = sample_td(
            value_function=self.critic,
            observation=observation,
            simulator=self.simulator,
            gamma=self.gamma,
            num_samples=self.num_samples,
            support=self.support,
            num_actions=self.num_actions,
        )

        dual = self.y_td[idx].detach() * td
        self._info.update(td=td)
        return dual.sum()


class REPSSaddleExactAgent(REPSExactAgent):
    """Saddle REPS agent."""

    def __init__(
        self,
        saddle_lr,
        saddle_mixing,
        lr,
        gamma,
        optimizer_=SGD,
        *args,
        **kwargs,
    ):
        super().__init__(lr=lr, gamma=gamma, *args, **kwargs)
        self.algorithm = REPSSaddleAlgorithm(
            saddle_lr=saddle_lr,
            saddle_mixing=saddle_mixing,
            gamma=self.gamma,
            *args,
            **kwargs,
        )
        self.policy = self.algorithm.policy
        self.optimizer = optimizer_(self.algorithm.critic.parameters(), lr=lr)
