"""Python Script Template."""
from abc import ABC

import torch
from rllib.environment.utilities import transitions2kernelreward


class AbstractSimulator(ABC):
    """Abstract simulator."""

    def __init__(self, no_simulator):
        self.no_simulator = no_simulator

    def query(self, observation, num_samples=1):
        """Query the simulator at a state action pair."""
        assert self.no_simulator
        next_state = observation.next_state.unsqueeze(0)
        reward = observation.reward.unsqueeze(0)

        if next_state.ndim > 2:
            next_state = next_state.squeeze(2)
        if reward.ndim > 2:
            reward = reward.squeeze(2)

        return next_state, reward

    @classmethod
    def default(cls, environment, no_simulator=False):
        """Get a simulator from the environment."""
        if hasattr(environment.env, "transitions"):
            return ExactSimulator.from_environment(environment, no_simulator)
        else:
            return EnvironmentSimulator.from_environment(environment, no_simulator)


class DatasetSimulator(AbstractSimulator):
    """Implementation of a from a dataset."""

    def __init__(self, experience_replay, no_simulator):
        super().__init__(no_simulator=no_simulator)
        self.experience_replay = experience_replay

    def query(self, observation, num_samples=1):
        """Query the simulator at a state action pair."""
        if self.no_simulator:
            return super().query(observation, num_samples)
        state, action = observation.state.squeeze(-1), observation.action.squeeze(-1)
        all_data = self.experience_replay.all_raw
        idx = torch.where(all_data.state == state + all_data.action == action)[0]
        assert len(idx) > 0
        i = idx[torch.randint(high=len(idx), size=())]
        return all_data.next_state[i], all_data.reward[i]


class ExactSimulator(AbstractSimulator):
    """Implementation of a simulator."""

    def __init__(self, transition_kernel, rewards, no_simulator):
        super().__init__(no_simulator=no_simulator)
        if not isinstance(transition_kernel, torch.Tensor):
            transition_kernel = torch.tensor(transition_kernel).float()
        if not isinstance(rewards, torch.Tensor):
            rewards = torch.tensor(rewards).float()

        self.transition_kernel = transition_kernel
        self.rewards = rewards

    @classmethod
    def from_environment(cls, environment, no_simulator):
        """Get simulator from an environment."""
        transition_kernel, reward = transitions2kernelreward(
            environment.env.transitions,
            num_states=environment.num_states,
            num_actions=environment.num_actions,
        )
        return cls(transition_kernel, reward, no_simulator)

    def query(self, observation, num_samples=1):
        """Query the simulator at a state action pair."""
        if self.no_simulator:
            return super().query(observation, num_samples)
        state, action = (
            observation.state.long().squeeze(-1),
            observation.action.long().squeeze(-1),
        )
        next_state_distribution = torch.distributions.Categorical(
            probs=self.transition_kernel[state, action]
        )
        next_state = next_state_distribution.sample((num_samples,))
        rewards = self.rewards[state, action].unsqueeze(0)
        rewards = rewards.repeat_interleave(num_samples, 0)
        return next_state, rewards


class EnvironmentSimulator(AbstractSimulator):
    """Implementation of a simulator."""

    def __init__(self, environment, no_simulator):
        super().__init__(no_simulator)
        self.environment = environment

    @classmethod
    def from_environment(cls, environment, no_simulator):
        """Get simulator from an environment."""
        return cls(environment, no_simulator)

    def query(self, observation, num_samples=1):
        """Query the simulator at a state action pair.

        TODO: states and actions will probably be tensors.
        TODO: Probably environment expects a single state not a batch.
        TODO: num_samples > 1?
        TODO: check if environment is vectorized?
        """
        if self.no_simulator:
            return super().query(observation, num_samples)
        self.environment.reset()
        self.environment.state = observation.state
        next_state, reward, done, info = self.environment.step(observation.action)
        return next_state, reward
