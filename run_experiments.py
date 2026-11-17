"""
Runs the experiments.

    python run_experiments.py              # 100 games per matchup
    python run_experiments.py --games 20   # quick smoke test

Every run writes into its own new folder, results/run_<date>_<time> (games.csv has one row
per game, summary.csv one row per matchup), so a smoke test never mixes with a real run.
Agents that are not available yet are skipped, so this can be run at any time.

NOTE: to plug in your agent, make it importable as shown in load_agents() (it needs
a .name and choose_move(board, rng), see the interface at the top of harness.py). If your
module exists but does not define the expected name, you get an error instead of a skip.
"""

import argparse
import time

from greedy import GreedyAgent
from harness import run_match

EVAL_SEED = 0       # seeds of the final evaluation games
TUNE_SEED = 1000    # use THIS base seed when SA tunes against Greedy, so the evaluation
                    # seeds stay unseen (the harness uses seeds base_seed ... base_seed + games - 1)

# (agent A, agent B) pairs; the names are the keys of load_agents()
PRIMARY = [("Minimax", "Greedy"), ("SA-Minimax", "Greedy"), ("TD", "Greedy")]
HEAD_TO_HEAD = [("Minimax", "TD"), ("SA-Minimax", "TD"), ("SA-Minimax", "Minimax")]


def load_minimax():
    from minimax import MinimaxAgent
    return MinimaxAgent(depth=3)


def load_sa_minimax():
    from sa_minimax import tuned_minimax_agent
    return tuned_minimax_agent()


def load_td():
    from td import TDAgent
    return TDAgent.load("td_weights.json")


def load_agents():
    """
    Builds every agent that is available.

    Returns:
    dict: Agent name -> agent object.
    """
    agents = {"Greedy": GreedyAgent()}
    loaders = {"Minimax": load_minimax, "SA-Minimax": load_sa_minimax, "TD": load_td}
    for name, loader in loaders.items():
        try:
            agents[name] = loader()
        except (ModuleNotFoundError, FileNotFoundError) as error:
            print(f"{name} is not available yet ({error})")
    return agents


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--games", type=int, default=100)
    parser.add_argument("--out", default=None, help="folder for the csv files (default: a new results/run_<date>_<time>)")
    args = parser.parse_args()
    out_dir = args.out or time.strftime("results/run_%Y%m%d_%H%M%S")
    print("Writing results to", out_dir)

    agents = load_agents()
    for name_a, name_b in PRIMARY + HEAD_TO_HEAD:
        if name_a in agents and name_b in agents:
            run_match(agents[name_a], agents[name_b], n_games=args.games,
                      base_seed=EVAL_SEED, out_dir=out_dir)
        else:
            print(f"Skipping {name_a} vs {name_b}")
