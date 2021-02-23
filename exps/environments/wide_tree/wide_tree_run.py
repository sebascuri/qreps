"""Python Script Template."""
from rllib.environment import GymEnvironment
from saddle_reps.environment.random_action_wrapper import RandomActionWrapper
from rllib.util.utilities import set_random_seed
import os

from exps.utilities import parse_arguments, run_experiment
from exps.environments.utilities import get_saddle_agents, get_benchmark_agents

args = parse_arguments()
args.env_name = "WideTree-v0"
args.lr = 0.05

set_random_seed(args.seed)
env = GymEnvironment(args.env_name, seed=args.seed)
env.add_wrapper(RandomActionWrapper, p=args.random_action_p)

agents = get_saddle_agents(env, **vars(args))
agents.update(get_benchmark_agents(env, **vars(args)))

df = run_experiment(agents, env, args)
df.to_pickle(f"wide_tree_results_{args.seed}.pkl")

# os.system("python wide_tree_plot.py")
