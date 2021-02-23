"""Python Script Template."""
import matplotlib.pyplot as plt


def plot_logs(logs):
    """Plot saddle logs."""
    plt.plot(logs["return"])
    plt.title("Total Returns")
    plt.xlabel("Num Episode")
    plt.show()

    fig, ax = plt.subplots(2, 1, sharex="row")
    ax[0].plot(logs["td"], "b", label="TD = r + V(x') - Q(x, a)")
    ax[0].plot(logs["adv"], "r", label="Advantage = Q(x, a) - V(x)")
    ax[0].set_ylabel("TD/Advantage")
    ax[0].legend()

    ax[1].plot(logs["saddle"])
    ax[1].set_ylabel("Saddle Objective")
    ax[1].set_xlabel("Num Iteration")
    plt.show()

    fig, ax = plt.subplots(1, 1, sharex="row")
    ax.plot(logs["entropy_y_td"], label="td entropy")
    ax.plot(logs["entropy_y_adv"], label="adv entropy")
    ax.legend()

    ax.set_ylabel("Entropy")
    ax.set_xlabel("Num Iteration")
    plt.show()
