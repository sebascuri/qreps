"""Python Script Template."""
from rllib.environment import GymEnvironment
from saddle_reps.environment.random_action_wrapper import RandomActionWrapper

import os

from exps.utilities import parse_arguments, run_experiment
from exps.action_gap.utilities import get_alpha_agents

args = parse_arguments()
args.seed = 1
args.max_steps = 50
args.batch_size = 50
args.num_iter = 50

env = GymEnvironment(args.env_name)
env.add_wrapper(RandomActionWrapper, p=args.random_action_p)

agents = get_alpha_agents(env, **vars(args))

df = run_experiment(agents, env, args)
df.to_pickle(f"action_gap_results.pkl")

os.system("python action_gap_plot.py")
