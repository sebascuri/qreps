"""Python Script Template."""
import seaborn as sns
from exps.utilities import (
    get_exact_reps,
    get_exact_q_reps,
    get_saddle_q_reps,
    get_sample_reps,
)

palette = sns.color_palette(n_colors=10)


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
        "SaddleQREPS-ER": get_saddle_q_reps(
            env, eta=eta / 2, alpha=eta / 2, no_simulator=True, *args, **kwargs
        ),
        "SaddleQREPS": get_saddle_q_reps(
            env, eta=eta / 2, alpha=eta / 2, *args, **kwargs
        ),
    }
    return agents


def get_biased_agents(env, eta, num_samples, *args, **kwargs):
    """Return agents that need the model."""
    agents = {
        "Biased-REPS": get_sample_reps(env, eta=eta, num_samples=0, *args, **kwargs),
        "Stochastic-REPS": get_sample_reps(
            env, eta=eta, num_samples=1, *args, **kwargs
        ),
        "Mini-Batch-REPS": get_sample_reps(
            env, eta=eta, num_samples=num_samples, *args, **kwargs
        ),
    }
    return agents


def get_linestyle(name: str):
    """Get agent linestyle."""
    if "-ER" in name or "Biased" in name or "Stochastic" in name or "Mini" in name:
        return "solid"
    elif "Saddle" in name:
        return "dotted"
    else:
        return "dashed"


def get_color(name: str):
    """Get plot color."""
    if "QREPS" in name:
        return palette[0]
    elif "Biased" in name:
        return palette[2]
    elif "Stochastic" in name:
        return palette[1]
    elif "Mini" in name:
        return palette[3]
    else:
        return palette[1]
