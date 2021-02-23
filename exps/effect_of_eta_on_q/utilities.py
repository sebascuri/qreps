"""Python Script Template."""
import seaborn as sns
from exps.utilities import get_exact_reps, get_exact_q_reps, get_sample_qreps

palette = sns.color_palette(n_colors=10)


"""Python Script Template."""
import seaborn as sns
from exps.utilities import get_exact_reps, get_exact_q_reps

palette = sns.color_palette(n_colors=10)


def get_eta_agents(env, eta, alpha, *args, **kwargs):
    """Return agents that need the model."""
    agents = {
        "ExactQREPS-0.001": get_exact_q_reps(
            env, eta=0.001 * eta, alpha=1.0, *args, **kwargs
        ),
        "ExactQREPS-0.01": get_exact_q_reps(
            env, eta=0.01 * eta, alpha=1.0, *args, **kwargs
        ),
        "ExactQREPS-0.1": get_exact_q_reps(
            env, eta=0.1 * eta, alpha=1.0, *args, **kwargs
        ),
        "ExactQREPS-1": get_exact_q_reps(env, eta=eta, alpha=1.0, *args, **kwargs),
        "ExactQREPS-10": get_exact_q_reps(
            env, eta=10 * eta, alpha=1.0, *args, **kwargs
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
    if "-100" in name:
        return palette[1]
    elif "-10" in name:
        return palette[2]
    elif "-1" in name:
        return palette[0]
    elif "-0.1" in name:
        return palette[3]
    elif "-0.01" in name:
        return palette[4]
    elif "-0.001" in name:
        return palette[8]
