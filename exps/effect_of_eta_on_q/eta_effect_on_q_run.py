"""Python Script Template."""
from rllib.environment import GymEnvironment
from qreps.environment.random_action_wrapper import RandomActionWrapper

import os

from exps.utilities import parse_arguments, run_experiment
from exps.effect_of_eta_on_q.utilities import get_eta_agents

args = parse_arguments()
args.num_episodes = 10
env = GymEnvironment(args.env_name)
env.add_wrapper(RandomActionWrapper, p=args.random_action_p)

agents = get_eta_agents(env, **vars(args))

df = run_experiment(agents, env, args)
df.to_pickle(f"eta_on_q_results.pkl")

# os.system("python eta_effect_plot.py")
