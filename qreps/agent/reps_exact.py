"""Python Script Template."""

import torch
from rllib.agent.off_policy.off_policy_agent import \
    OffPolicyAgent  # Has memory.
from rllib.algorithms.abstract_algorithm import AbstractAlgorithm
from rllib.dataset.datatypes import Loss
from rllib.environment.utilities import transitions2kernelreward
from torch.optim import SGD

from qreps.algorithms.simulator import AbstractSimulator
from qreps.policy.exact_reps_policy import ExactREPSPolicy
from qreps.util.accumulative_module import AccumulativeModule
from qreps.util.utilities import (build_empirical_mu, build_empirical_y0,
                                        compute_exact_td,
                                        get_default_value_function,
                                        get_primal_reps)


class REPSExactAlgorithm(AbstractAlgorithm):
    """Implementation of REPS Algorithm.

    It uses a model to compute the policy.
    It uses a model to compute an unbiased estimate of the dual.
    """

    def __init__(
        self,
        eta,
        critic,
        transitions,
        rewards,
        num_actions,
        gamma=1.0,
        support="state-action",
        *args,
        **kwargs,
    ):
        critic = AccumulativeModule(critic)
        policy = ExactREPSPolicy(
            value_function=critic,
            eta=eta,
            transitions=transitions,
            rewards=rewards,
            gamma=gamma,
            num_actions=num_actions,
        )
        super().__init__(gamma=gamma, policy=policy, critic=critic)
        self.transitions = transitions
        self.rewards = rewards

        self.eta = eta
        self.num_states = self.policy.num_states
        self.num_actions = self.policy.num_actions

        self.y0 = torch.zeros(1)
        if self.num_states > 0:
            self.prior = torch.zeros(self.num_states, self.num_actions)
        self.support = support

    def init(self, observation):
        """Initialize algorithm."""
        self.y0 = build_empirical_y0(
            observation=observation, support=self.support, policy=self.old_policy
        )
        if self.num_states > 0:
            self.prior = build_empirical_mu(
                observation=observation,
                support=self.support,
                policy=self.old_policy,
            )

    def update_multipliers(self, idx):
        """Update lagrange multipliers."""
        pass

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

    def get_discount_dual_loss(self, state, idx):
        """Get loss in the dual that appears due to discounting."""
        y0 = self.y0[idx]
        if self.support == "state":
            y0 = y0.sum(-1)
        return ((1 - self.gamma) * y0 * self.critic(state)).sum()

    def dual(self, observation, idx):
        """Get dual, td, and advantage."""
        td = compute_exact_td(
            value_function=self.critic,
            observation=observation,
            transitions=self.transitions,
            rewards=self.rewards,
            gamma=self.gamma,
            support=self.support,
        )
        dual = self.y0[idx] * torch.exp(self.eta * td)
        return torch.log(dual.sum()) / self.eta

    def forward(self, observation, idx=None):
        """Compute losses at state/idx pairs."""
        state = observation.state
        if idx is None:
            idx = torch.arange(state.shape[0])
        return Loss(
            dual_loss=self.dual(observation, idx=idx)
            + self.get_discount_dual_loss(state, idx)
        )

    def reset(self):
        """Reset episode."""
        super().reset()
        try:
            self.critic.reset()
        except AttributeError:
            pass


class REPSExactAgent(OffPolicyAgent):
    """Exact REPS agent."""

    def __init__(
        self,
        critic,
        eta,
        transitions,
        rewards,
        num_iter,
        lr,
        gamma,
        support,
        num_rollouts=1,
        train_frequency=0,
        reset_memory_after_learn=True,
        eval_distribution=None,
        optimizer_=SGD,
        *args,
        **kwargs,
    ):
        super().__init__(
            num_iter=num_iter,
            num_rollouts=num_rollouts,
            train_frequency=train_frequency,
            reset_memory_after_learn=reset_memory_after_learn,
            gamma=gamma,
            *args,
            **kwargs,
        )
        self.algorithm = REPSExactAlgorithm(
            eta=eta,
            critic=critic,
            transitions=transitions,
            rewards=rewards,
            support=support,
            gamma=self.gamma,
            *args,
            **kwargs,
        )
        self.policy = self.algorithm.policy
        self.optimizer = optimizer_(self.algorithm.critic.parameters(), lr=lr)
        self.transitions = transitions
        self.rewards = rewards
        self.eval_distribution = eval_distribution

    def learn(self):
        """Learn algorithm."""
        all_observations = self.memory.all_data
        idx_ = torch.arange(all_observations.state.shape[0])
        self.algorithm.init(all_observations)

        def closure(observation=all_observations, idx=idx_):
            """Gradient calculation."""
            # TODO: What happens if we sample a mini-batch instead of the full-batch?
            self.optimizer.zero_grad()
            if self.batch_size == len(self.memory):
                observation, idx, _ = self.memory.sample_batch(self.batch_size)
            losses = self.algorithm(observation, idx=idx)
            self.optimizer.zero_grad()
            losses.dual_loss.backward()
            torch.nn.utils.clip_grad_norm_(
                (p for p in self.algorithm.parameters() if p.requires_grad),
                self.clip_gradient_val,
            )
            self.algorithm.update_multipliers(idx)
            return losses

        self._learn_steps(closure)
        if self.reset_memory_after_learn:
            self.memory.reset()

    @classmethod
    def default(
        cls,
        environment,
        num_iter=100,
        eta=1.0,
        gamma=1.0,
        simulator=None,
        no_simulator=False,
        function_approximation="tabular",
        *args,
        **kwargs,
    ):
        """Get default agent."""
        if simulator is None:
            simulator = AbstractSimulator.default(environment, no_simulator)

        try:
            transitions, rewards = transitions2kernelreward(
                environment.env.transitions,
                environment.num_states,
                environment.num_actions,
            )
            transitions = torch.tensor(transitions).float()
            rewards = torch.tensor(rewards).float()
        except AttributeError:
            transitions, rewards = None, None
        return super().default(
            environment,
            num_actions=environment.num_actions,
            eta=eta,
            num_iter=num_iter,
            critic=get_default_value_function(environment, function_approximation),
            transitions=transitions,
            rewards=rewards,
            gamma=gamma,
            simulator=simulator,
            *args,
            **kwargs,
        )
