"""Python Script Template."""
from rllib.environment import GymEnvironment
from saddle_reps.environment.random_action_wrapper import RandomActionWrapper
from rllib.util.utilities import set_random_seed
import os

from exps.utilities import parse_arguments, run_experiment
from exps.environments.utilities import get_saddle_agents, get_benchmark_agents

args = parse_arguments()
args.env_name = "WindyGrid-v0"
args.lr = 0.03
args.eta = 1.0

set_random_seed(args.seed)
env = GymEnvironment(args.env_name, seed=args.seed)
env.add_wrapper(RandomActionWrapper, p=args.random_action_p)

agents = get_saddle_agents(env, **vars(args))
agents.update(get_benchmark_agents(env, **vars(args)))
for param_group in agents["REINFORCE"].optimizer.param_groups:
    param_group["lr"] = 0.001

df = run_experiment(agents, env, args)
df.to_pickle(f"windy_grid_world_results_{args.seed}.pkl")
