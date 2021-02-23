"""Python Script Template."""
import seaborn as sns
from exps.utilities import get_exact_reps, get_exact_q_reps

palette = sns.color_palette(n_colors=10)


def get_alpha_agents(env, eta, alpha, *args, **kwargs):
    """Return agents that need the model."""
    agents = {
        "ExactREPS-1": get_exact_reps(env, eta=eta, *args, **kwargs),
        "ExactQREPS-1": get_exact_q_reps(
            env, eta=eta / 2, alpha=alpha / 2, *args, **kwargs
        ),
        "ExactQREPS-3": get_exact_q_reps(
            env, eta=eta / 2, alpha=3 * eta / 2, *args, **kwargs
        ),
        "ExactQREPS-10": get_exact_q_reps(
            env, eta=eta / 2, alpha=10 * eta / 2, *args, **kwargs
        ),
        "ExactQREPS-0.3": get_exact_q_reps(
            env, eta=eta / 2, alpha=0.3 * eta / 2, *args, **kwargs
        ),
        "ExactQREPS-0.1": get_exact_q_reps(
            env, eta=eta / 2, alpha=0.1 * eta, *args, **kwargs
        ),
    }
    return agents


def get_linestyle(name: str):
    """Get agent linestyle."""
    if "QREPS" in name:
        return "solid"
    else:
        return "dashed"


def get_color(name: str):
    """Get plot color."""
    if "-10" in name:
        return palette[4]
    elif "-3" in name:
        return palette[3]
    elif "-1" in name:
        return palette[0]
    elif "-0.3" in name:
        return palette[2]
    elif "-0.1" in name:
        return palette[1]
    else:
        return palette[8]
