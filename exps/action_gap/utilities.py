"""Python Script Template."""
import seaborn as sns
from exps.utilities import get_saddle_q_reps, get_exact_q_reps

palette = sns.color_palette(n_colors=10)


def get_alpha_agents(env, eta, alpha, lr, *args, **kwargs):
    """Return agents that need the model."""
    agents = {
        "ExactQREPS-0.1": get_exact_q_reps(
            env, eta=1.0, alpha=0.1, lr=1.0, *args, **kwargs
        ),
        "ExactQREPS-1": get_exact_q_reps(
            env, eta=1.0, alpha=1.0, lr=0.1, *args, **kwargs
        ),
        "ExactQREPS-10": get_exact_q_reps(
            env, eta=1.0, alpha=10.0, lr=0.1, *args, **kwargs
        ),
        "ExactQREPS-100": get_exact_q_reps(
            env, eta=1.0, alpha=100.0, lr=0.1, *args, **kwargs
        ),
        "ExactQREPS-1000": get_exact_q_reps(
            env, eta=10.0, alpha=1000.0, lr=0.1, *args, **kwargs
        ),
        "QREPS-0.1": get_saddle_q_reps(
            env, eta=0.1, alpha=0.1, lr=0.1, *args, **kwargs
        ),
        "QREPS-1": get_saddle_q_reps(env, eta=1.0, alpha=1.0, lr=0.1, *args, **kwargs),
        "QREPS-10": get_saddle_q_reps(
            env, eta=1.0, alpha=10.0, lr=0.1, *args, **kwargs
        ),
        "QREPS-100": get_saddle_q_reps(
            env, eta=1.0, alpha=100.0, lr=0.1, *args, **kwargs
        ),
        "QREPS-1000": get_saddle_q_reps(
            env, eta=10.0, alpha=1000.0, lr=0.1, *args, **kwargs
        ),
    }
    return agents


def get_linestyle(name: str):
    """Get agent linestyle."""
    if "Exact" in name:
        return "dashed"
    else:
        return "solid"


def get_color(name: str):
    """Get plot color."""
    if "-1000" in name:
        return palette[1]
    elif "-100" in name:
        return palette[2]
    elif "-10" in name:
        return palette[3]
    elif "-1" in name:
        return palette[0]
    elif "-0.1" in name:
        return palette[4]
    elif "-0.01" in name:
        return palette[5]
    else:
        return palette[8]
