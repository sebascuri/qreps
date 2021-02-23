import torch
from torch.nn.modules.loss import MSELoss, SmoothL1Loss
from rllib.policy import TabularPolicy
from torch.distributions import Categorical
from saddle_reps.util.utilities import average_policy_evaluation, mdp2mrp
import matplotlib.pyplot as plt
import seaborn as sns

from exps.plotting import set_figure_params

palette = sns.color_palette(n_colors=15)


eta = 0.5
num_states = 3
num_actions = 2
P = torch.tensor(
    [
        [[0.8, 0.2, 0.0], [0.05, 0.9, 0.05]],
        [[0.0, 1.0, 0.0], [0.05, 0.05, 0.9]],
        [[0.1, 0.1, 0.8], [0.05, 0.05, 0.9]],
    ]
)
r = torch.tensor([[-0.0, 0], [-0.1, 0], [-0.0, 1.0]])
pi = Categorical(probs=torch.tensor([[0.0, 1.0], [0.0, 1.0], [0.0, 1.0]]))
policy = TabularPolicy(num_states=num_states, num_actions=num_actions)
for state in range(num_states):
    policy.set_value(state, pi.logits[state])

rho = average_policy_evaluation(policy=policy, transitions=P, rewards=r)
P_, r_ = mdp2mrp(policy=policy, transitions=P, rewards=r)

r = r - rho
r_ = r_ - rho

V = torch.zeros(num_states)
for i in range(1000):
    V[1:] = P_[1:, 0, :] @ V + r_[1:, 0]

# Q = r + (P @ V).squeeze(-1)
Q = torch.tensor([[-1.0, 0], [0.5, 1.0], [0.2, 0.5]])

Q_ = Q.clone()
dqn_error = torch.zeros(100)
dqn_huber = torch.zeros(100)
log_error = torch.zeros(100)
q20 = torch.linspace(-2.4, 2.5, 100)

alpha = 1.2
for i, q in enumerate(q20):
    Q_[2, 0] = q
    pred = Q_.reshape(-1)
    target = (r + (P @ Q_.max(-1)[0])).reshape(-1)
    dqn_error[i] = MSELoss(reduction="mean")(pred, target)
    dqn_huber[i] = SmoothL1Loss(reduction="mean")(alpha * pred, alpha * target)

    V_ = 1 / eta * (torch.logsumexp(eta * Q_, dim=-1))
    td = r + (P @ V_).squeeze(-1) - Q_
    Z = torch.log(torch.tensor(1.0 * num_states * num_actions))
    log_error[i] = 1 / eta * (torch.logsumexp(eta * td, dim=(0, 1)) - Z)

dqn_error -= dqn_error.min()
dqn_huber -= dqn_huber.min()
log_error -= log_error.min()

set_figure_params(serif=True, fontsize=12)
fig, ax = plt.subplots(ncols=1, nrows=1)
fig.set_size_inches(6.75 / 2, 1.5)
# WIDTH = \columnwidth = 6.75 / 2in
plt.plot(
    q20.detach(),
    log_error.detach().numpy(),
    label=r"\textbf{Logistic}",
    color=palette[0],
    linestyle="solid",
)
plt.plot(
    q20.detach(),
    dqn_error.detach().numpy(),
    label="Squared",
    color=palette[1],
    linestyle="dashed",
)
plt.plot(
    q20.detach(),
    dqn_huber.detach().numpy(),
    label="Huber",
    color=palette[2],
    linestyle="dotted",
)
plt.ylabel("Bellman Error")
plt.xlabel("$Q(x=0, a=0)$")
ax.set_xticklabels({})
ax.set_yticklabels({})
plt.legend(loc="best", frameon=False)
fig.tight_layout(pad=0.2)
plt.savefig("bellman_error.pdf")
