"""Python Script Template."""
from lsf_runner import init_runner, make_commands
import os
import numpy as np
import time

cwd = os.path.dirname(os.path.realpath(__file__))

ENVIRONMENTS = [
    "cart_pole",
    "double_chain",
    "river_swim",
    "single_chain",
    "two_state_deterministic",
    "two_state_stochastic",
    "wide_tree",
    "windy_grid_world",
]
for environment in ENVIRONMENTS:
    runner = init_runner(f"QREPS_{environment}", num_threads=2)
    script = f"{environment}/{environment}_run.py"
    commands = make_commands(
        f"{cwd}/{script}",
        base_args={"env_name": environment},
        common_hyper_args={"seed": np.arange(50)},
    )
    runner.run_batch(commands)
    time.sleep(1)
