"""Python Script Template."""
from rllib.environment import GymEnvironment
from saddle_reps.environment.random_action_wrapper import RandomActionWrapper

import os

from exps.utilities import parse_arguments, run_experiment
from exps.effect_of_alpha.utilities import get_alpha_agents

args = parse_arguments()

env = GymEnvironment(args.env_name)
env.add_wrapper(RandomActionWrapper, p=args.random_action_p)

agents = get_alpha_agents(env, **vars(args))

df = run_experiment(agents, env, args)
df.to_pickle(f"alpha_results.pkl")

os.system("python alpha_effect_plot.py")
