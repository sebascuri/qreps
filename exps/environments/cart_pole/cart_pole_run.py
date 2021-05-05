"""Python Script Template."""
from rllib.environment import GymEnvironment
from qreps.environment.random_action_wrapper import RandomActionWrapper
from rllib.util.utilities import set_random_seed
import os
from torch.optim import Adam
from exps.utilities import parse_arguments, run_experiment
from qreps.agent.q_reps_agent import QREPSAgent
from exps.environments.utilities import get_benchmark_agents

args = parse_arguments()
args.env_name = "CartPole-v0"
args.eta = 1 / 100.0
args.alpha = 1 / 100.0
args.lr = 0.08
args.num_episodes = 25
args.function_approximation = "linear"
args.num_rollouts = 4
args.num_iter = 200
args.gamma = 0.99
weight_decay = 1e-4
args.optimizer_ = Adam
args.seed = 0

set_random_seed(args.seed)
env = GymEnvironment(args.env_name, args.seed)
env.add_wrapper(RandomActionWrapper, p=args.random_action_p)

agent = QREPSAgent.default(env, **vars(args), weight_decay=weight_decay)
agent.batch_size = 1
agents = {"SaddleQREPS": agent}
# agents.update(get_benchmark_agents(env, **vars(args)))

df = run_experiment(agents, env, args, 0)
import matplotlib.pyplot as plt

plt.plot(agents["SaddleQREPS"].logger.all["dual_loss"])
plt.show()

# df.to_pickle(f"cart_pole_results_{args.seed}.pkl")
df.to_pickle(f"cart_pole_results.pkl")
os.system("python cart_pole_plot.py")
