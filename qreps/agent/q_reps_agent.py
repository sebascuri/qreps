"""Implementation of Q-REPS Agent."""

from rllib.agent import REPSAgent

from qreps.algorithms.q_reps import QREPS
from qreps.util.utilities import get_default_q_function
from qreps.util.accumulative_module import AccumulativeModule
from torch.optim import SGD


class QREPSAgent(REPSAgent):
    """Implementation of the REPS algorithm.

    References
    ----------
    Peters, J., Mulling, K., & Altun, Y. (2010, July).
    Relative entropy policy search. AAAI.

    Deisenroth, M. P., Neumann, G., & Peters, J. (2013).
    A survey on policy search for robotics. Foundations and Trends® in Robotics.
    """

    def __init__(
        self,
        policy,
        critic,
        q_function,
        eta=1.0,
        regularization=True,
        learn_policy=False,
        weight_decay=1e-3,
        minibatch=False,
        *args,
        **kwargs,
    ):
        super().__init__(policy=policy, critic=critic, *args, **kwargs)
        self.algorithm = QREPS(
            q_function=q_function,
            eta=eta,
            regularization=regularization,
            learn_policy=learn_policy,
            *args,
            **kwargs,
        )
        # Over-write optimizer.
        optimizer = kwargs.get("optimizer_", SGD)
        self.optimizer = optimizer(
            [
                p
                for name, p in self.algorithm.named_parameters()
                if "target" not in name
            ],
            lr=self.optimizer.defaults["lr"],
            weight_decay=weight_decay,
        )
        self.policy = self.algorithm.policy
        self.minibatch = minibatch

    def learn(self):
        """Optimize dual"""
        if self.minibatch:
            super().learn()
            return
        observation = self.memory.all_data

        def closure():
            """Gradient calculation."""
            self.optimizer.zero_grad()
            losses = self.algorithm(observation.clone())
            self.optimizer.zero_grad()
            losses.dual_loss.backward()

            return losses

        self._learn_steps(closure)

        if self.reset_memory_after_learn:
            self.memory.reset()  # Erase memory.

    @classmethod
    def default(cls, environment, function_approximation="linear", *args, **kwargs):
        """See `AbstractAgent.default'."""
        q_function = get_default_q_function(environment, function_approximation)
        if function_approximation in ["linear", "tabular"]:
            q_function = AccumulativeModule(q_function)
        return super().default(environment, q_function=q_function, *args, **kwargs)
