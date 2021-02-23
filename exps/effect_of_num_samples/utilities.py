"""Python Script Template."""
import seaborn as sns
from exps.utilities import (
    get_exact_reps,
    get_exact_q_reps,
    get_saddle_q_reps,
    get_sample_reps,
)

palette = sns.color_palette(n_colors=15)


def get_exact_agents(env, eta, alpha, *args, **kwargs):
    """Return agents that need the model."""
    agents = {
        "ExactREPS": get_exact_reps(env, eta=eta, *args, **kwargs),
    }
    return agents


def get_biased_agents(env, eta, num_samples, *args, **kwargs):
    """Return agents that need the model."""
    agents = {
        "BiasedREPS": get_sample_reps(env, eta=eta, num_samples=0, *args, **kwargs),
        "SampleREPS-1": get_sample_reps(env, eta=eta, num_samples=1, *args, **kwargs),
        "SampleREPS-2": get_sample_reps(env, eta=eta, num_samples=2, *args, **kwargs),
        "SampleREPS-5": get_sample_reps(env, eta=eta, num_samples=5, *args, **kwargs),
        "SampleREPS-10": get_sample_reps(env, eta=eta, num_samples=10, *args, **kwargs),
    }
    return agents


def get_linestyle(name: str):
    """Get agent linestyle."""
    if "Sample" in name:
        return "solid"
    elif "Exact" in name:
        return "dashed"
    else:
        return "dotted"


def get_color(name: str):
    """Get plot color."""
    if "ExactREPS" in name:
        return palette[1]
    elif "SampleREPS-10" in name:
        return palette[4]
    elif "SampleREPS-1" in name:
        return palette[1]
    elif "SampleREPS-2" in name:
        return palette[2]
    elif "SampleREPS-5" in name:
        return palette[3]
    elif "BiasedREPS" in name:
        return palette[6]
    else:
        return palette[0]
