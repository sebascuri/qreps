"""Implementation of REPS Algorithm."""

import torch
import torch.distributions
from rllib.algorithms.reps import REPS
from rllib.dataset.datatypes import Loss

from qreps.value_function.soft_value_function import SoftValueFunction
from qreps.policy.q_reps_policy import QREPSPolicy


class QREPS(REPS):
    r"""Q-Relative Entropy Policy Search Algorithm.

    Q-REPS optimizes the following regularized LP over the vectors
    \mu(X, A) and d(X, A).

    ..math::  \max \mu r - eta R(\mu, d)
    ..math::  s.t. \sum_a d(x, a) = \sum_{x', a'} = \mu(x', a') P(x|x', a'),
    ..math::  s.t.  d(x, a) = \mu(x, a)
    ..math::  s.t.  1/2 (\mu + d) is a distribution.

    The LP dual is:
    ..math::  G(V) = \eta \log \sum_{x, a} d(x, a) 0.5 (
                \exp^{\delta(x, a) / \eta} + \exp^{(Q(x, a) - V(x)) / \eta}).


    where \delta(x,a) = r + \sum_{x'} P(x'|x, a) V(x') - V(x) is the TD-error and V(x)
    are the dual variables associated with the stationary constraints in the primal,
    and Q(x, a) are the dual variables associated with the equality of distributions
    constraint.
    V(x) is usually referred to as the value function and Q(x, a) as the q-function.

    Using d(x,a) as the empirical distribution, G(V) can be approximated by samples.

    The optimal policy is given by:
    ..math::  \pi(a|x) \propto d(x, a) \exp^{Q(x, a) / \eta}.

    By default, the policy is a soft-max policy.
    However, if Q-REPS is initialized with a parametric policy it can be fit by
    minimizing the negative log-likelihood at the sampled elements.


    Calling REPS() returns a sampled based estimate of G(V) and the NLL of the policy.
    Both G(V) and NLL lend are differentiable and lend themselves to gradient based
    optimization.


    References
    ----------
    Peters, J., Mulling, K., & Altun, Y. (2010, July).
    Relative entropy policy search. AAAI.

    Deisenroth, M. P., Neumann, G., & Peters, J. (2013).
    A survey on policy search for robotics. Foundations and Trends® in Robotics.
    """

    def __init__(
        self, q_function, eta, alpha=None, learn_policy=False, *args, **kwargs
    ):
        kwargs.pop("critic", None)
        kwargs.pop("policy", None)
        if alpha is None:
            alpha = eta
        critic = SoftValueFunction(q_function, param=alpha)
        policy = QREPSPolicy(q_function, param=alpha)
        super().__init__(
            eta=eta,
            critic=critic,
            policy=policy,
            learn_policy=learn_policy,
            entropy_regularization=True,
            *args,
            **kwargs
        )
        self.q_function = q_function

    def actor_loss(self, observation):
        """Return primal and dual loss terms from Q-REPS."""
        state, action, reward, next_state, done, *r = observation

        # Calculate dual variables
        value = self.critic(state)
        target = self.get_value_target(observation)
        q_value = self.q_function(state, action)

        td = target - q_value
        self._info.update(td=td)

        # Calculate weights.
        weights_td = self.eta() * td  # type: torch.Tensor
        if weights_td.ndim == 1:
            weights_td = weights_td.unsqueeze(-1)
        dual = 1 / self.eta() * torch.logsumexp(weights_td, dim=-1)
        dual += (1 - self.gamma) * value.squeeze(-1)
        return Loss(dual_loss=dual.mean(), td_error=td)
