"""Python Script Template."""
import argparse
import pandas as pd
from rllib.util.rollout import rollout_agent
from rllib.util.utilities import set_random_seed
import numpy as np
from qreps.agent import (
    REPSExactAgent,
    REPSSampleExactAgent,
    REPSSaddleExactAgent,
    QREPSExactAgent,
    QREPSSaddleAgent,
    QREPSAgent,
)
from qreps.util.utilities import (
    average_policy_evaluation,
    linear_system_policy_evaluation,
)

from rllib.environment.utilities import transitions2kernelreward
import seaborn as sns
import torch
import torch.optim as optim

palette = sns.color_palette(n_colors=15)


def get_saddle_reps(env, *args, **kwargs):
    """Get saddle REPS agent."""
    return REPSSaddleExactAgent.default(env, *args, **kwargs)


def get_exact_reps(env, *args, **kwargs):
    """Get exact REPS agent."""
    return REPSExactAgent.default(env, *args, **kwargs)


def get_sample_reps(env, *args, **kwargs):
    """Get sample-version REPS agent."""
    return REPSSampleExactAgent.default(env, *args, **kwargs)


def get_sample_qreps(env, *args, **kwargs):
    """Get Sample-version of Q-REPS agent."""
    return QREPSAgent.default(env, *args, **kwargs)


def get_exact_q_reps(env, *args, **kwargs):
    """Get exact Q-REPS agent."""
    return QREPSExactAgent.default(env, *args, **kwargs)


def get_saddle_q_reps(env, *args, **kwargs):
    """Get Saddle Q-REPS agent."""
    return QREPSSaddleAgent.default(env, *args, **kwargs)


def get_qreps_agents(env, eta, alpha=None, *args, **kwargs):
    """Get Q-REPS agents. """
    agents = {
        "SaddleQREPS-ER": get_saddle_q_reps(
            env, eta=eta, no_simulator=True, alpha=alpha, *args, **kwargs
        ),
        "ExactQREPS": get_exact_q_reps(env, eta=eta, alpha=alpha, *args, **kwargs),
        "SaddleQREPS": get_saddle_q_reps(env, eta=eta, alpha=alpha, *args, **kwargs),
    }
    return agents


def get_er_agents(env, eta, alpha=None, *args, **kwargs):
    """Get experience replay agents."""
    agents = {
        "SaddleREPS-ER": get_saddle_reps(
            env, eta=eta, no_simulator=True, *args, **kwargs
        ),
        "Stochastic-REPS-ER": get_sample_reps(
            env, eta=eta, no_simulator=True, *args, **kwargs
        ),
        # "SaddleQVREPS-ER": get_saddle_q_reps(
        #     env, eta=2 * eta, no_simulator=True, *args, **kwargs
        # ),
        "SaddleQREPS-ER": get_saddle_q_reps(
            env, eta=eta, no_simulator=True, alpha=alpha, *args, **kwargs
        ),
    }
    return agents


def get_exact_agents(env, eta, alpha=None, *args, **kwargs):
    """Return agents that need the model."""
    agents = {
        "ExactREPS": get_exact_reps(env, eta=eta, *args, **kwargs),
        # "ExactQVREPS": _exact_q_reps(env, eta=2 * eta, *args, **kwargs),
        "ExactQREPS": get_exact_q_reps(env, eta=eta, alpha=alpha, *args, **kwargs),
    }
    return agents


def get_simulator_agents(env, eta, num_samples=1, alpha=None, *args, **kwargs):
    """Get agents that require a simulator."""
    agents = {
        "SaddleREPS": get_saddle_reps(env, eta=eta, *args, **kwargs),
        "Biased-REPS": get_sample_reps(env, eta=eta, num_samples=0, *args, **kwargs),
        "Stochastic-REPS": get_sample_reps(
            env, eta=eta, num_samples=1, *args, **kwargs
        ),
        "Mini-Batch-REPS": get_sample_reps(
            env, eta=eta, num_samples=num_samples, *args, **kwargs
        ),
        # "SaddleQVREPS": get_saddle_q_reps(
        #     env, eta=2 * eta, num_samples=num_samples, *args, **kwargs
        # ),
        "SaddleQREPS": get_saddle_q_reps(
            env, eta=eta, num_samples=num_samples, alpha=alpha, *args, **kwargs
        ),
    }
    return agents


def get_all_agents(env, *args, **kwargs):
    """Get all agents."""
    agents = dict()
    if kwargs.get("support", None) == "state-action":
        agents.update(**get_er_agents(env, *args, **kwargs))
    if hasattr(env.env, "transitions"):
        agents.update(**get_exact_agents(env, *args, **kwargs))

    agents.update(**get_simulator_agents(env, *args, **kwargs))
    return agents


def get_linestyle(name: str):
    """Get agent linestyle."""
    if name.startswith("Exact"):
        return "dashed"
    elif name.endswith("ER") or "REPS" not in name:
        return "solid"
    else:
        return "dotted"


def get_color(name: str):
    """Get plot color."""
    return {
        "SaddleREPS-ER": palette[0],
        "SaddleQVREPS-ER": palette[1],
        "SaddleQREPS-ER": palette[2],
        "ExactREPS": palette[0],
        "ExactQVREPS": palette[1],
        "ExactQREPS": palette[2],
        "SaddleREPS": palette[0],
        "SaddleQVREPS": palette[1],
        "SaddleQREPS": palette[2],
        "Stochastic-REPS-ER": palette[3],
        "Stochastic-REPS": palette[3],
        "Mini-Batch-REPS": palette[4],
        "Biased-REPS": palette[5],
        "DQN-delayed": palette[6],
        "DQN-polyak": palette[7],
        "PPO": palette[8],
        "VMPO": palette[9],
        "REINFORCE": palette[10],
        "ParametricREPS-ER": palette[0],
        "ExactQREPS-0.1": palette[1],
        "ExactQREPS-0.3": palette[2],
        "ExactQREPS-0.5": palette[3],
        "ExactQREPS-1": palette[4],
        "ExactQREPS-3": palette[5],
        "ExactQREPS-5": palette[6],
        "ExactQREPS-10": palette[7],
    }.get(name, palette[np.random.choice(len(palette))])


def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--env_name", type=str, default="TwoStateStochastic-v0")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--num_episodes", type=int, default=10)
    parser.add_argument("--max_steps", type=int, default=200)
    parser.add_argument("--batch_size", type=int, default=200)
    parser.add_argument("--num_iter", type=int, default=300)
    parser.add_argument("--num_rollouts", type=int, default=1)
    parser.add_argument("--eta", type=float, default=1.0)
    parser.add_argument("--alpha", type=float, default=1.0)
    parser.add_argument("--lr", type=float, default=0.1)
    parser.add_argument("--saddle_lr", type=float, default=0.1)
    parser.add_argument("--saddle_mixing", type=float, default=1e-4)
    parser.add_argument("--random_action_p", type=float, default=0)

    parser.add_argument("--num_samples", type=int, default=10)
    parser.add_argument("--gamma", type=float, default=1.0)
    parser.add_argument(
        "--support", type=str, default="state-action", choices=["state-action", "state"]
    )
    parser.add_argument(
        "--function_approximation",
        type=str,
        default="tabular",
        choices=["tabular", "linear", "neural_network"],
    )
    parser.add_argument("--optimizer_", type=str, default="SGD")

    args = parser.parse_args()
    args.optimizer_ = getattr(optim, args.optimizer_)
    return args


def evaluate_policy(agent, environment, _=None):
    """Evaluate policy with callback."""
    try:
        transitions, rewards = transitions2kernelreward(
            environment.env.transitions,
            environment.num_states,
            environment.num_actions,
        )
        transitions = torch.tensor(transitions).float()
        rewards = torch.tensor(rewards).float()

    except AttributeError:
        return
    if agent.gamma == 1:
        value = average_policy_evaluation(agent.policy, transitions, rewards)
    else:
        vf = linear_system_policy_evaluation(
            agent.policy, transitions, rewards, gamma=agent.gamma
        )
        values = vf.table.detach()[:, :-1] * (1 - agent.gamma)
        if agent.eval_distribution is None:
            value = values.mean()
        else:
            value = (agent.eval_distribution * values).sum()
    agent.logger.update(value_evaluation=value)


def store_value_function(agent, environment=None, episode=None):
    """Store value function."""
    try:
        v = agent.algorithm.critic.func.table
        agent.logger.update(v_0=v[0, 0])
        agent.logger.update(v_1=v[0, 1])

        vbar = agent.algorithm.critic.running_func.table
        agent.logger.update(vbar_0=vbar[0, 0])
        agent.logger.update(vbar_1=vbar[0, 1])
    except AttributeError:
        pass

    try:
        q = agent.algorithm.q_function.func.table
        agent.logger.update(q_00=q[0, 0])
        agent.logger.update(q_01=q[1, 0])
        agent.logger.update(q_10=q[0, 1])
        agent.logger.update(q_11=q[1, 1])

        qbar = agent.algorithm.q_function.running_func.table
        agent.logger.update(qbar_00=qbar[0, 0])
        agent.logger.update(qbar_01=qbar[1, 0])
        agent.logger.update(qbar_10=qbar[0, 1])
        agent.logger.update(qbar_11=qbar[1, 1])
    except AttributeError:
        pass


def run_experiment(agents, environment, args, seed=None):
    """Run a set of experiments."""
    seed = args.seed if seed is None else seed
    df = pd.DataFrame()
    for name, agent in agents.items():
        set_random_seed(seed)
        print(f"Running agent {name} on {args.env_name}")
        evaluate_policy(agent, environment)
        store_value_function(agent, environment, episode=0)

        rollout_agent(
            agent=agent,
            environment=environment,
            num_episodes=args.num_episodes,
            max_steps=args.max_steps,
            callback_frequency=1,
            callbacks=[evaluate_policy, store_value_function],
        )

        df_ = pd.DataFrame(agent.logger.statistics)
        df_["name"] = name
        df_["seed"] = args.seed
        df_["time"] = np.arange(len(df_))
        df_["duals"] = np.empty((len(df_)), dtype=object)
        duals = agent.logger.all["dual_loss"]
        for i in range(len(df_)):
            df_.at[i, "duals"] = duals[i * args.num_iter : (i + 1) * args.num_iter]
        df = pd.concat((df, df_), sort=False)
    return df
