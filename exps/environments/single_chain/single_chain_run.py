"""Python Script Template."""
from rllib.environment import GymEnvironment
from rllib.util.utilities import set_random_seed
from qreps.environment.random_action_wrapper import RandomActionWrapper

import os

from exps.utilities import parse_arguments, run_experiment
from exps.environments.utilities import get_saddle_agents, get_benchmark_agents

args = parse_arguments()
args.env_name = "SingleChainProblem-v0"
args.eta = 10.0
args.lr = 0.05

set_random_seed(args.seed)
env = GymEnvironment(args.env_name, seed=args.seed)
env.add_wrapper(RandomActionWrapper, p=args.random_action_p)

agents = get_saddle_agents(env, **vars(args))
agents.update(get_benchmark_agents(env, **vars(args)))

df = run_experiment(agents, env, args)
df.to_pickle(f"single_chain_results_{args.seed}.pkl")

# os.system("python single_chain_plot.py")
