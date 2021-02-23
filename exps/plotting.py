"""Python Script Template."""
import matplotlib.pyplot as plt
from matplotlib.colors import ColorConverter
from matplotlib import rcParams
import numpy as np


def plot_logger_key(
    agents, key, title_string, save_string, y_label, get_color, get_linestyle,
):
    """Plot logger keys."""
    plt.close()
    plt.clf()
    plt.xlabel("Episode")
    plt.ylabel(y_label)
    plt.title(title_string)
    for name, agent in agents.items():
        values = agent.logger.get(key)
        plt.plot(
            np.arange(len(values)),
            values,
            label=name,
            linestyle=get_linestyle(name),
            color=get_color(name),
        )
    plt.legend(bbox_to_anchor=(1.05, 1), loc="upper left", frameon=False)
    plt.savefig(save_string, bbox_inches="tight")


def plot_df_key(df, key, axes, get_color, get_linestyle, max_value=1, saturate=True):
    """Plot df from a key."""
    for name, value in df.groupby("name"):
        value = value[["time", "seed", key]]
        mean = value.groupby("time")[key].mean() / max_value
        std = value.groupby("time")[key].std() / max_value

        upper = mean + std
        lower = mean - std
        if saturate:
            upper[upper > 1] = 1
            lower[lower < 0] = 0

        color = get_color(name)

        if name == "SaddleQREPS":
            label = r"\textbf{Q-REPS}"
        elif "DQN" in name:
            label = "DQN"
        else:
            label = name

        axes.plot(
            mean, label=label, linestyle=get_linestyle(name), color=color,
        )
        if len(value.seed.unique()) > 1:
            axes.fill_between(
                np.arange(len(mean)), lower, upper, color=color, alpha=0.2,
            )


def emulate_color(color, alpha=1, background_color=(1, 1, 1)):
    """Take an RGBA color and an RGB background, return the emulated RGB color.

    The RGBA color with transparency alpha is converted to an RGB color via
    emulation in front of the background_color.
    """
    to_rgb = ColorConverter().to_rgb
    color = to_rgb(color)
    background_color = to_rgb(background_color)
    return [
        (1 - alpha) * bg_col + alpha * col
        for col, bg_col in zip(color, background_color)
    ]


def cm2inches(centimeters):
    """Convert cm to inches"""
    return centimeters / 2.54


def set_figure_params(serif=True, fontsize=9):
    """Define default values for font, fontsize and use latex

    Parameters
    ----------
    serif: bool, optional
        Whether to use a serif or sans-serif font
    """
    preamble = r"\DeclareMathAlphabet{\mathcal}{OMS}{cmsy}{m}{n} \usepackage{newtxmath}"
    params = {
        "font.serif": [
            "Times",
            "Palatino",
            "New Century Schoolbook",
            "Bookman",
            "Computer Modern Roman",
        ]
        + rcParams["font.serif"],
        "font.sans-serif": [
            "Times",
            "Helvetica",
            "Avant Garde",
            "Computer Modern Sans serif",
        ]
        + rcParams["font.sans-serif"],
        "font.family": "serif",
        "text.usetex": True,
        # Make sure mathcal doesn't use the Times style
        "text.latex.preamble": preamble,
        "axes.labelsize": fontsize,
        "axes.linewidth": 0.75,
        "font.size": fontsize,
        "legend.fontsize": fontsize,
        "xtick.labelsize": fontsize * 8 / 9,
        "ytick.labelsize": fontsize * 8 / 9,
        # 'figure.dpi': 150,
        # 'savefig.dpi': 600,
        "legend.numpoints": 1,
    }

    if not serif:
        params["font.family"] = "sans-serif"

    rcParams.update(params)


def hide_all_ticks(axis):
    """Hide all ticks on the axis.

    Parameters
    ----------
    axis: matplotlib axis
    """
    axis.tick_params(
        axis="both",  # changes apply to the x-axis
        which="both",  # affect both major and minor ticks
        bottom=False,  # ticks along the bottom edge are off
        top=False,  # ticks along the top edge are off
        left=False,  # No ticks left
        right=False,  # No ticks right
        labelbottom=False,  # No tick-label at bottom
        labelleft=False,
    )  # No tick-label at bottom


def hide_spines(axis, top=True, right=True):
    """Hide the top and right spine of the axis."""
    if top:
        axis.spines["top"].set_visible(False)
        axis.xaxis.set_ticks_position("bottom")
    if right:
        axis.spines["right"].set_visible(False)
        axis.yaxis.set_ticks_position("left")


def set_frame_properties(axis, color, lw):
    """Set color and linewidth of frame."""
    for spine in axis.spines.values():
        spine.set_linewidth(lw)
        spine.set_color(color)


def linewidth_in_data_units(linewidth, axis, reference="y"):
    """
    Convert a linewidth in data units to linewidth in points.

    Parameters
    ----------
    linewidth: float
        Linewidth in data units of the respective reference-axis
    axis: matplotlib axis
        The axis which is used to extract the relevant transformation
        data (data limits and size must not change afterwards)
    reference: string
        The axis that is taken as a reference for the data width.
        Possible values: 'x' and 'y'. Defaults to 'y'.

    Returns
    -------
    linewidth: float
        Linewidth in points
    """
    fig = axis.get_figure()

    if reference == "x":
        # width of the axis in inches
        axis_length = fig.get_figwidth() * axis.get_position().width
        value_range = np.diff(axis.get_xlim())
    elif reference == "y":
        axis_length = fig.get_figheight() * axis.get_position().height
        value_range = np.diff(axis.get_ylim())

    # Convert axis_length from inches to points
    axis_length *= 72

    return (linewidth / value_range) * axis_length


def adapt_figure_size_from_axes(axes):
    """
    Adapt the figure sizes so that all axes are equally wide/high.

    When putting multiple figures next to each other in Latex, some
    figures will have axis labels, while others do not. As a result,
    having the same figure width for all figures looks really strange.
    This script adapts the figure sizes post-plotting, so that all the axes
    have the same width and height.

    Be sure to call plt.tight_layout() again after this operation!

    This doesn't work if you have multiple axis on one figure and want them
    all to scale proportionally, but should be an easy extension.

    Parameters
    ----------
    axes: list
        List of axes that we want to have the same size (need to be
        on different figures)
    """
    # Get parent figures
    figures = [axis.get_figure() for axis in axes]

    # get axis sizes [0, 1] and figure sizes [inches]
    axis_sizes = np.array([axis.get_position().size for axis in axes])
    figure_sizes = np.array([figure.get_size_inches() for figure in figures])

    # Compute average axis size [inches]
    avg_axis_size = np.average(axis_sizes * figure_sizes, axis=0)

    # New figure size is the average axis size plus the white space that is
    # not begin used by the axis so far (e.g., the space used by labels)
    new_figure_sizes = (1 - axis_sizes) * figure_sizes + avg_axis_size

    # Set new figure sizes
    for figure, size in zip(figures, new_figure_sizes):
        figure.set_size_inches(size)
