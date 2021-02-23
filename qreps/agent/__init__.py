from rllib.agent import REPSAgent

from .q_reps_agent import QREPSAgent
from .q_reps_exact import QREPSExactAgent
from .q_reps_saddle import QREPSSaddleAgent
from .reps_exact import REPSExactAgent
from .reps_saddle_exact_policy import REPSSaddleExactAgent
from .reps_sample_exact_policy import REPSSampleExactAgent

AGENTS = [
    QREPSAgent,
    REPSAgent,
    REPSExactAgent,
    REPSSaddleExactAgent,
    REPSSampleExactAgent,
    QREPSExactAgent,
    QREPSSaddleAgent,
]
