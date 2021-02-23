"""Python Script Template."""
from itertools import product

import torch
from rllib.algorithms.tabular_planning.utilities import init_value_function
from rllib.dataset.datatypes import Observation
from rllib.policy import NNPolicy, TabularPolicy
from rllib.util.utilities import tensor_to_distribution
from rllib.value_function import (
    NNQFunction,
    NNValueFunction,
    TabularQFunction,
    TabularValueFunction,
)


def freeze_hidden_layers(module):
    """Freeze the hidden layers of a module.

    Parameters
    ----------
    module : torch.nn.Module
    """
    for name, param in module.named_parameters():
        if "hidden_layers" in name:
            param.requires_grad = False


def build_empirical_y0(observation, support, policy=None):
    """Build empirical distribution over samples."""
    state = observation.state
    num_states_ = state.shape[0]
    if support == "state-action":
        y0 = torch.ones(num_states_) / float(num_states_)
    elif support == "state":
        pi = tensor_to_distribution(policy(state).detach())
        y0 = pi.probs / float(num_states_)
    else:
        raise NotImplementedError(f"{support} not implemented.")
    return y0


def build_empirical_mu(observation, support, policy):
    """Build empirical mu distribution."""
    num_states, num_actions = policy.num_states, policy.num_actions
    prior = torch.zeros(num_states, num_actions)
    state, action = observation.state.long(), observation.action.long()
    num_states_ = state.shape[0]
    if support == "state-action":
        for i_s, i_a in product(range(num_states), range(num_actions)):
            prior[i_s, i_a] = ((i_s == state) & (i_a == action)).sum()
        eps = 1.0 / num_states
        prior = (prior.float() + eps) / (num_states_ + 1.0)
    elif support == "state":
        for i_s in range(num_states):
            i_s = torch.tensor(i_s).long()
            pi = tensor_to_distribution(policy(i_s).detach())
            count = (i_s == state).sum()
            prior[i_s] = pi.probs * count / float(num_states_)
    else:
        raise NotImplementedError(f"{support} not implemented.")
    return prior


def _compute_exact_td(
    value_function, observation, transitions, rewards, gamma,
):
    """Compute exact TD at a given state-action pair."""
    state, action = observation.state.long(), observation.action.long()
    if state.ndim > 2:
        state, action = state.squeeze(1), action.squeeze(1)

    num_states, num_actions = rewards.shape
    next_states = torch.arange(num_states)

    rewards_ = rewards[state, action]
    probability = transitions[state, action]
    pred = value_function(state)
    td = rewards_ + gamma * probability @ value_function(next_states) - pred
    if td.ndim > 1:
        td = td.squeeze(1)
    return td


def compute_exact_td(value_function, observation, transitions, rewards, gamma, support):
    """Compute exact TD at a given state/action pairs."""
    if transitions is None:
        return compute_sample_td(
            value_function,
            observation.state,
            observation.reward,
            observation.next_state,
            gamma,
            observation.done,
        )
    if support == "state-action":
        td = _compute_exact_td(value_function, observation, transitions, rewards, gamma)
    elif support == "state":
        state = observation.state
        if state.ndim > 0:
            num_states = state.shape[0]
            if state.ndim > 1:
                state = state.squeeze(1)
        else:
            num_states = 1
        num_actions = rewards.shape[1]
        td = torch.zeros(num_states, num_actions)
        for action in range(num_actions):
            action = torch.tensor(action)
            observation.action = action
            td[:, action] = _compute_exact_td(
                value_function, observation, transitions, rewards, gamma
            )
        if state.ndim == 0:
            td = td[0]
    else:
        raise NotImplementedError(f"{support} not implemented.")
    return td


def sample_td(
    value_function,
    observation,
    gamma,
    support,
    simulator=None,
    num_samples=1,
    num_actions=1,
):
    """Sample td."""
    state = observation.state
    if state.ndim > 1:
        state = state.squeeze(1)
    if support == "state-action":
        next_state, reward = simulator.query(observation, num_samples=num_samples)
        if simulator.no_simulator:
            done = observation.done.unsqueeze(0)
            if done.ndim > 2:
                done = done.squeeze(2)
        else:
            done = 0
        td = compute_sample_td(
            value_function, state, reward, next_state, gamma, done=done
        ).mean(0)
    elif support == "state":
        if state.ndim > 0:
            num_states = state.shape[0]
        else:
            num_states = 1
        td = torch.zeros(num_states, num_actions)
        for action in range(num_actions):
            action = torch.tensor(action)
            next_state, reward = simulator.query(observation, num_samples=num_samples)
            td[:, action] = compute_sample_td(
                value_function, state, reward, next_state, gamma, done=0
            ).mean(0)
        if state.ndim == 0:
            td = td[0]
    else:
        raise NotImplementedError(f"{support} not implemented.")
    return td


def compute_expected_etd(
    value_function, observation, transitions, rewards, gamma, eta, support,
):
    r"""Compute expected e^{eta * td}."""
    state, action = observation.state.long(), observation.action.long()
    num_states_ = state.shape[0]
    num_states, num_actions = rewards.shape
    if support == "state-action":
        probabilities = transitions[state, action]
        rewards_ = rewards[state, action]
        td = compute_sample_td(
            value_function,
            state.unsqueeze(-1),
            rewards_.unsqueeze(-1),
            torch.arange(num_states).unsqueeze(0).repeat_interleave(num_states_, 0),
            gamma,
            done=0,
        )
        expected_etd = (torch.exp(eta * td) * probabilities).sum(-1)

    elif support == "state":
        expected_etd = torch.zeros(num_states_, num_actions)
        for action in range(num_actions):
            action = torch.tensor(action)
            probabilities = transitions[state, action]
            rewards_ = rewards[state, action]
            td = compute_sample_td(
                value_function,
                state.unsqueeze(-1),
                rewards_.unsqueeze(-1),
                torch.arange(num_states).unsqueeze(0).repeat_interleave(num_states_, 0),
                gamma,
                done=0,
            )
            expected_etd[..., action] = (torch.exp(eta * td) * probabilities).sum(-1)
    else:
        raise NotImplementedError(f"{support} not implemented.")
    return expected_etd


def compute_sample_td(value_function, state, reward, next_state, gamma, done):
    """Compute sampled TD at a given state-action pair."""
    return (
        reward + gamma * value_function(next_state) * (1 - done) - value_function(state)
    )


def compute_advantage(value_function, q_function, state):
    """Compute advantage at a given state-action pair."""
    return q_function(state) - value_function(state).unsqueeze(-1)


def get_primal_reps(eta, value_function, transitions, rewards, gamma, prior=None):
    """Get Primal for REPS."""
    num_states, num_actions = rewards.shape
    td = torch.zeros(num_states, num_actions)
    states = torch.arange(num_states)
    for action in range(num_actions):
        action = torch.tensor(action)
        observation = Observation(state=states, action=action)
        td[:, action] = _compute_exact_td(
            value_function, observation, transitions, rewards, gamma,
        )

    max_td = torch.max(eta * td)
    if prior is not None:
        mu_log = torch.log(prior * torch.exp(eta * td - max_td)) + max_td
        mu_dist = torch.distributions.Categorical(logits=mu_log.reshape(-1))
    else:
        mu_dist = torch.distributions.Categorical(logits=(eta * td).reshape(-1))
    mu = mu_dist.probs.reshape(num_states, num_actions)
    return mu


def get_primal_q_reps(
    prior, eta, value_function, q_function, transitions, rewards, gamma
):
    """Get Primal for Q-REPS."""
    num_states, num_actions = prior.shape
    td = torch.zeros(num_states, num_actions)
    states = torch.arange(num_states)
    for action in range(num_actions):
        action = torch.tensor(action)
        observation = Observation(state=states, action=action)
        td[:, action] = _compute_exact_td(
            value_function, observation, transitions, rewards, gamma,
        )

    td = td + value_function(states).unsqueeze(-1) - q_function(states)
    max_td = torch.max(eta * td)

    mu_log = torch.log(prior * torch.exp(eta * td - max_td)) + max_td
    mu_dist = torch.distributions.Categorical(logits=mu_log.reshape(-1))
    mu = mu_dist.probs.reshape(num_states, num_actions)

    adv = q_function(states) - value_function(states).unsqueeze(-1)
    max_adv = torch.max(eta * adv)
    d_log = torch.log(prior * torch.exp(eta * adv - max_adv)) + max_adv
    d_dist = torch.distributions.Categorical(logits=d_log.reshape(-1))
    d = d_dist.probs.reshape(num_states, num_actions)

    return mu, d


def mdp2mrp(transitions, rewards, policy, terminal_states=None):
    """Transform MDP and Policy to an MRP.

    Parameters
    ----------
    transitions: Tensor.
    rewards: Tensor.
    policy: AbstractPolicy.

    Returns
    -------
    environment: MDP.
    """
    num_states, num_actions = rewards.shape
    mrp_kernel = torch.zeros((num_states, 1, num_states))
    mrp_reward = torch.zeros((num_states, 1))

    if terminal_states is None:
        terminal_states = []

    for state in range(num_states):
        if state in terminal_states:
            mrp_kernel[state, 0, state] = 1
            mrp_reward[state] = 0
            continue

        state = torch.tensor(state).long()
        policy_ = tensor_to_distribution(policy(state), **policy.dist_params)

        for action, p_action in enumerate(policy_.probs):
            for next_state, p_next_state in enumerate(transitions[state, action]):
                mrp_reward[state, 0] += p_action * p_next_state * rewards[state, action]
                mrp_kernel[state, 0, next_state] += p_action * p_next_state

    return mrp_kernel, mrp_reward


def linear_system_policy_evaluation(
    policy, transitions, rewards, gamma, value_function=None
):
    """Evaluate a policy in an MDP solving the system bellman of equations.

    V = r + gamma * P * V
    V = (I - gamma * P)^-1 r
    """
    num_states, num_actions = rewards.shape
    if value_function is None:
        value_function = init_value_function(num_states, terminal_states=None)

    mrp_transitions, mrp_rewards = mdp2mrp(transitions, rewards, policy)

    A = torch.eye(num_states) - gamma * mrp_transitions[:, 0, :]
    vals = A.inverse() @ mrp_rewards[:, 0]
    for state in range(num_states):
        value_function.set_value(state, vals[state].item())

    return value_function


def average_policy_evaluation(policy, transitions, rewards):
    r"""Evaluate policy.

    Finds stationary distribution (right eigenvector of 1 eigenvalue) and computes
    ..math:: \langle \mu, r \rangle.

    """
    mrp_transitions, mrp_rewards = mdp2mrp(transitions, rewards, policy)
    eig_values, eig_vectors = torch.eig(mrp_transitions[:, 0, :].T, eigenvectors=True)
    idx = torch.where(torch.isclose(eig_values, torch.tensor([1.0, 0.0])).all(-1))
    idx = idx[0]
    try:
        stationary_distribution = eig_vectors[:, idx] / torch.sum(eig_vectors[:, idx])
        return (stationary_distribution.T @ mrp_rewards).item()
    except ValueError:
        return 0


def accumulate_parameters(target_module, new_module, count):
    """Accumulate the parameters of target_target_module with those of new_module.

    The parameters of target_nn are replaced by:
        target_params <- (count * target_params + new_params) / (count + 1)

    Parameters
    ----------
    target_module: nn.Module
    new_module: nn.Module
    count: int.

    Returns
    -------
    None.
    """
    with torch.no_grad():
        target_state_dict = target_module.state_dict()
        new_state_dict = new_module.state_dict()

        for name in target_state_dict.keys():
            if target_state_dict[name] is new_state_dict[name]:
                continue
            else:
                if target_state_dict[name].data.ndim == 0:
                    target_state_dict[name].data = new_state_dict[name].data
                else:
                    target_state_dict[name].data[:] = (
                        count * target_state_dict[name].data + new_state_dict[name].data
                    ) / (count + 1)


def get_default_q_function(environment, function_approximation):
    """Get default Q-Function."""
    if function_approximation == "tabular":
        q_function = TabularQFunction.default(environment)
    elif function_approximation == "linear":
        q_function = NNQFunction.default(environment, layers=[200])
        freeze_hidden_layers(q_function)
    else:
        q_function = NNQFunction.default(environment)
    return q_function


def get_default_value_function(environment, function_approximation):
    """Get default Value Function."""
    if function_approximation == "tabular":
        value_function = TabularValueFunction.default(environment)
    elif function_approximation == "linear":
        value_function = NNValueFunction.default(environment, layers=[200])
        freeze_hidden_layers(value_function)
    else:
        value_function = NNValueFunction.default(environment)
    return value_function


def get_default_policy(environment, function_approximation):
    """Get default policy."""
    if function_approximation == "tabular":
        policy = TabularPolicy.default(environment)
    elif function_approximation == "linear":
        policy = NNPolicy.default(environment, layers=[200])
        freeze_hidden_layers(policy)
    else:
        policy = NNPolicy.default(environment)
    return policy
