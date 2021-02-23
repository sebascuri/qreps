"""Python Script Template."""
from itertools import chain
import seaborn as sns

from qreps.util.utilities import (
    get_default_q_function,
    get_default_policy,
    get_default_value_function,
)
from rllib.agent import DQNAgent, REPSAgent, PPOAgent, VMPOAgent, REINFORCEAgent
from rllib.dataset import ExperienceReplay
from rllib.policy import EpsGreedy
from rllib.util.parameter_decay import ExponentialDecay

from exps.utilities import get_saddle_q_reps, get_exact_reps, get_exact_q_reps

palette = sns.color_palette(n_colors=15)


def get_dqn(
    env,
    lr,
    optimizer_,
    num_rollouts,
    num_iter,
    tau,
    target_update_frequency,
    function_approximation="tabular",
    *args,
    **kwargs,
):
    """Get DQN agent."""
    critic = get_default_q_function(env, function_approximation)
    critic.tau = tau
    policy = EpsGreedy(critic, ExponentialDecay(0.1, 0, 1000))
    optimizer = optimizer_(critic.parameters(), lr=lr)
    memory = ExperienceReplay(max_len=50000, num_steps=0)
    return DQNAgent(
        critic=critic,
        policy=policy,
        optimizer=optimizer,
        memory=memory,
        reset_memory_after_learn=True,
        train_frequency=0,
        num_iter=num_iter,
        target_update_frequency=target_update_frequency,
        num_rollouts=num_rollouts,
        *args,
        **kwargs,
    )


def get_reps_parametric(
    env, optimizer_, lr, function_approximation="tabular", *args, **kwargs
):
    """Get Parametric REPS agent."""
    critic = get_default_value_function(env, function_approximation)
    policy = get_default_policy(env, function_approximation)
    optimizer = optimizer_(chain(critic.parameters(), policy.parameters()), lr=lr)
    memory = ExperienceReplay(max_len=50000, num_steps=0)
    return REPSAgent(
        critic=critic,
        policy=policy,
        optimizer=optimizer,
        memory=memory,
        *args,
        **kwargs,
    )


def get_ppo(env, optimizer_, lr, function_approximation="tabular", *args, **kwargs):
    """Get PPO agent."""
    critic = get_default_value_function(env, function_approximation)
    policy = get_default_policy(env, function_approximation)
    optimizer = optimizer_(chain(critic.parameters(), policy.parameters()), lr)
    return PPOAgent(critic=critic, policy=policy, optimizer=optimizer, *args, **kwargs)


def get_vmpo(env, optimizer_, lr, function_approximation="tabular", *args, **kwargs):
    """Get VMPO agent."""
    critic = get_default_value_function(env, function_approximation)
    policy = get_default_policy(env, function_approximation)
    memory = ExperienceReplay(max_len=50000, num_steps=0)
    optimizer = optimizer_(chain(critic.parameters(), policy.parameters()), lr)
    return VMPOAgent(
        policy=policy,
        critic=critic,
        optimizer=optimizer,
        memory=memory,
        *args,
        **kwargs,
    )


def get_reinforce(
    env, optimizer_, lr, function_approximation="tabular", *args, **kwargs
):
    """Get REINFORCE agent."""
    critic = get_default_value_function(env, function_approximation)
    policy = get_default_policy(env, function_approximation)
    optimizer = optimizer_(
        [
            {"params": policy.parameters(), "lr": lr / 3},
            {"params": critic.parameters(), "lr": lr},
        ]
    )
    return REINFORCEAgent(
        policy=policy, critic=critic, optimizer=optimizer, *args, **kwargs
    )


def get_benchmark_agents(env, eta, *args, **kwargs):
    """Get benchmark agents."""
    agents = {
        "DQN-delayed": get_dqn(
            env, tau=0, target_update_frequency=kwargs.get("num_iter"), *args, **kwargs
        ),
        "DQN-polyak": get_dqn(
            env, tau=5e-3, target_update_frequency=1, *args, **kwargs
        ),
        "PPO": get_ppo(env, *args, **kwargs),
        "VMPO": get_vmpo(env, *args, **kwargs),
        "REINFORCE": get_reinforce(env, *args, **kwargs),
        "REPS": get_reps_parametric(env, eta=eta, *args, **kwargs),
    }
    return agents


def get_exact_agents(env, eta, alpha, *args, **kwargs):
    """Return agents that need the model."""
    agents = {
        "ExactREPS": get_exact_reps(env, eta=eta, *args, **kwargs),
        "ExactQREPS": get_exact_q_reps(
            env, eta=eta / 2, alpha=eta / 2, *args, **kwargs
        ),
    }
    return agents


def get_saddle_agents(env, eta, alpha, *args, **kwargs):
    """Return agents that need the model."""
    agents = {
        "SaddleQREPS": get_saddle_q_reps(
            env, eta=eta / 2, alpha=eta / 2, no_simulator=True, *args, **kwargs
        ),
    }
    return agents


def get_linestyle(name):
    """Get linestyle."""
    if "QREPS" in name:
        return "solid"
    elif "REPS" in name and "REPS" != name:
        return "solid"
    else:
        return {
            "DQN-delayed": "dashed",
            "DQN-polyak": "dashed",
            "PPO": "dotted",
            "VMPO": "dashdot",
            "REINFORCE": (0, (3, 5, 1, 5, 1, 5)),
            "REPS": (0, (3, 5, 1, 5)),
        }[name]


def get_color(name):
    """Get line color."""
    if "QREPS" in name:
        return palette[0]
    elif "REPS" in name and "REPS" != name:
        return palette[1]
    else:
        return {
            "DQN-delayed": palette[2],
            "DQN-polyak": palette[3],
            "PPO": palette[4],
            "VMPO": palette[5],
            "REINFORCE": palette[6],
            "REPS": palette[7],
        }[name]
