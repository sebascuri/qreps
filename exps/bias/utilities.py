"""Python Script Template."""
import seaborn as sns
from exps.utilities import get_saddle_q_reps, get_exact_q_reps

palette = sns.color_palette(n_colors=10)


def get_eta_agents(env, eta, alpha, saddle_lr, *args, **kwargs):
    """Return agents that need the model."""
    agents = {
        "ExactQREPS-1": get_exact_q_reps(env, eta=1.0, alpha=1.0, *args, **kwargs),
        "SampleQREPS-1": get_saddle_q_reps(
            env, eta=1.0, alpha=1.0, saddle_lr=0.1, *args, **kwargs
        ),
        "ExactQREPS-10": get_exact_q_reps(env, eta=10, alpha=1.0, *args, **kwargs),
        "SampleQREPS-10": get_saddle_q_reps(
            env, eta=10, alpha=1.0, saddle_lr=0.1, *args, **kwargs
        ),
        "ExactQREPS-0.1": get_exact_q_reps(env, eta=0.1, alpha=1.0, *args, **kwargs),
        "SampleQREPS-0.1": get_saddle_q_reps(
            env, eta=0.1, alpha=1.0, saddle_lr=1.0, *args, **kwargs
        ),
    }
    return agents


def get_linestyle(name: str):
    """Get agent linestyle."""
    if "Sample" in name:
        return "solid"
    elif "QREPS" in name:
        return "dashed"
    else:
        return "dotted"


def get_color(name: str):
    """Get plot color."""
    if "-10" in name:
        return palette[6]
    elif "-1" in name:
        return palette[0]
    elif "-0.1" in name:
        return palette[7]
    else:
        return palette[9]
