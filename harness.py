"""
Evaluation harness: plays matches between two agents, logs every game, and summarizes
win rate (95% confidence interval), time per move and nodes per move.

Agent interface (what Greedy, Minimax, SA-Minimax and TD must provide):
    agent.name                       str shown in the result files. Results are grouped by it,
                                     so make it identify the settings, e.g. "Minimax(d=3)"
    agent.new_game()                 optional, called before every game
    agent.choose_move(board, rng)    returns a (src, dst) move from get_valid_moves(board).
                                     rng is a seeded random.Random: use it for ALL
                                     randomness (tie-breaking) so games can be replayed.
    agent.last_nodes                 optional int, nodes expanded for the last move

Fairness
  * Agent A moves first in the even-numbered games and agent B in the odd-numbered
    ones, so each side moves first in half of the games (use an even n_games).
  * Game g uses seed base_seed + g. Use different base seeds for SA tuning and for the
    final evaluation (run_experiments.py has TUNE_SEED and EVAL_SEED).
  * Every game gives each agent a fresh seeded rng, so any game can be replayed exactly.
    As a safety net the global random module is also seeded during a game (and restored
    afterwards), in case an agent calls random.choice like the connect four minimax does.
    numpy's random generator is NOT seeded, so use rng instead.
  * An agent that returns an illegal move stops the match with an error.
  * The moves of every game are logged (move_list), so a game can be replayed from the csv.

Usage:
    summary = run_match(MinimaxAgent(depth=3), GreedyAgent(), n_games=100, base_seed=0)
    # inside an SA tuning loop (no files written, no printing):
    summary = run_match(candidate, GreedyAgent(), n_games=20, base_seed=TUNE_SEED,
                        out_dir=None, verbose=False)
"""

import csv
import math
import os
import random
import time

from engine import BOARD_SIDE, create_board, get_valid_moves, make_move, is_terminal_node, total_distance


###############################################################################
#                                 Playing games                               #
###############################################################################

def play_game(agent_a, agent_b, a_first, seed, side=BOARD_SIDE):
    """
    Plays one game and measures both agents.

    Parameters:
    agent_a (object): Agent "A" (see the interface above).
    agent_b (object): Agent "B".
    a_first (bool): True if A is player 0 (moves first), False if B moves first.
    seed (int): Seed of this game. A and B each get their own rng derived from it.
    side (int): Board size (see engine.py).

    Returns:
    dict: winner ("A", "B" or "draw"), reason ("goal" or "cap"), plies, first_mover,
    both final total distances, each agent's number of moves, seconds and nodes, and
    move_list (every move as "src-dst", in the order played, player 0 first).
    """
    board = create_board(side)
    a_player = 0 if a_first else 1  # the player number that agent A plays
    b_player = 1 - a_player
    agents = {a_player: agent_a, b_player: agent_b}
    rngs = {a_player: random.Random(2 * seed), b_player: random.Random(2 * seed + 1)}

    num_moves = [0, 0]  # these three lists are indexed by player number, not by agent
    seconds = [0.0, 0.0]
    nodes = [0, 0]
    move_list = []

    for agent in (agent_a, agent_b):
        if hasattr(agent, "new_game"):
            agent.new_game()

    outside_state = random.getstate()  # see "Fairness" above
    random.seed(seed)
    try:
        while not is_terminal_node(board):
            player = board.turn
            agent = agents[player]

            start = time.perf_counter()
            move = agent.choose_move(board, rngs[player])
            seconds[player] += time.perf_counter() - start

            if move is None or tuple(move) not in get_valid_moves(board):
                raise ValueError(f"{agent.name} played an illegal move: {move}")
            move = tuple(move)  # a list such as [3, 7] is fine too
            move_list.append(f"{move[0]}-{move[1]}")
            num_moves[player] += 1
            nodes[player] += getattr(agent, "last_nodes", 0) or 0
            board = make_move(board, move)
    finally:
        random.setstate(outside_state)

    if board.winner is None:
        winner = "draw"
    elif board.winner == a_player:
        winner = "A"
    else:
        winner = "B"

    return {
        "winner": winner,
        "reason": board.end_reason,
        "plies": board.plies,
        "first_mover": "A" if a_first else "B",
        "a_final_dist": total_distance(board, a_player),
        "b_final_dist": total_distance(board, b_player),
        "a_moves": num_moves[a_player], "b_moves": num_moves[b_player],
        "a_time_s": seconds[a_player], "b_time_s": seconds[b_player],
        "a_nodes": nodes[a_player], "b_nodes": nodes[b_player],
        "move_list": " ".join(move_list),
    }


def run_match(agent_a, agent_b, n_games=100, base_seed=0, side=BOARD_SIDE,
              out_dir="results", tag="", verbose=True):
    """
    Plays n_games between A and B (alternating who moves first), appends each game to
    <out_dir>/games.csv as soon as it is finished (so a crash keeps the finished games)
    and one summary row to <out_dir>/summary.csv at the end.

    Parameters:
    agent_a (object): Agent "A".
    agent_b (object): Agent "B".
    n_games (int): Number of games to play (use an even number).
    base_seed (int): Seed of game 0; game g uses base_seed + g.
    side (int): Board size (see engine.py).
    out_dir (str): Folder for the csv files (created if needed), or None to write
    nothing, e.g. inside an SA tuning loop. Files are only ever appended to, so use a
    new folder for every experiment (run_experiments.py does this).
    tag (str): Optional label added to the matchup name, e.g. "tuning".
    verbose (bool): Print the summary when done.

    Returns:
    dict: The summary (same fields as the row written to summary.csv).
    """
    matchup = f"{agent_a.name} vs {agent_b.name}" + (f" [{tag}]" if tag else "")

    games_path = None
    if out_dir is not None:
        os.makedirs(out_dir, exist_ok=True)
        games_path = os.path.join(out_dir, "games.csv")

    rows = []
    for g in range(n_games):
        result = play_game(agent_a, agent_b, a_first=(g % 2 == 0), seed=base_seed + g, side=side)
        row = {"matchup": matchup, "game": g, "seed": base_seed + g,
               "a_name": agent_a.name, "b_name": agent_b.name}
        row.update(result)
        rows.append(row)
        if games_path is not None:
            append_rows(games_path, [row])

    summary = summarize(rows, matchup, base_seed)
    if out_dir is not None:
        append_rows(os.path.join(out_dir, "summary.csv"), [summary])

    if verbose:
        print_summary(summary)
    return summary


###############################################################################
#                           Summaries and csv output                          #
###############################################################################

def wilson_ci(wins, n, z=1.96):
    """
    Wilson score confidence interval for a win rate (z = 1.96 gives 95%).

    Parameters:
    wins (int): Number of wins.
    n (int): Number of games.
    z (float): z-score of the interval.

    Returns:
    tuple: (low, high) as fractions between 0 and 1.
    """
    if n == 0:
        return (0.0, 0.0)
    p = wins / n
    denominator = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / denominator
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denominator
    low = 0.0 if wins == 0 else max(0.0, centre - half)  # exactly 0 / 1 in these cases,
    high = 1.0 if wins == n else min(1.0, centre + half)  # floating point would give 0.99999...
    return (low, high)


def summarize(rows, matchup, base_seed):
    """
    Turns the per-game rows of one match into one summary dictionary (from A's side).

    Parameters:
    rows (list): The dictionaries returned by play_game.
    matchup (str): Name of the matchup.
    base_seed (int): Seed of the first game.

    Returns:
    dict: Wins, A's win rate with its 95% interval, A's wins (and games) as first and as
    second mover, average ms and nodes per move for each agent, average game length and
    the number of games ended by the cap.
    """
    n = len(rows)
    a_wins = sum(1 for r in rows if r["winner"] == "A")
    b_wins = sum(1 for r in rows if r["winner"] == "B")
    low, high = wilson_ci(a_wins, n)

    a_first_rows = [r for r in rows if r["first_mover"] == "A"]
    b_first_rows = [r for r in rows if r["first_mover"] == "B"]

    a_moves = max(1, sum(r["a_moves"] for r in rows))
    b_moves = max(1, sum(r["b_moves"] for r in rows))
    return {
        "matchup": matchup, "base_seed": base_seed, "games": n,
        "a_wins": a_wins, "b_wins": b_wins, "draws": n - a_wins - b_wins,
        "a_win_rate": a_wins / n, "a_ci_lo": low, "a_ci_hi": high,
        "a_wins_as_first": sum(1 for r in a_first_rows if r["winner"] == "A"),
        "a_games_as_first": len(a_first_rows),
        "a_wins_as_second": sum(1 for r in b_first_rows if r["winner"] == "A"),
        "a_games_as_second": len(b_first_rows),
        "a_ms_per_move": 1000 * sum(r["a_time_s"] for r in rows) / a_moves,
        "b_ms_per_move": 1000 * sum(r["b_time_s"] for r in rows) / b_moves,
        "a_nodes_per_move": sum(r["a_nodes"] for r in rows) / a_moves,
        "b_nodes_per_move": sum(r["b_nodes"] for r in rows) / b_moves,
        "avg_plies": sum(r["plies"] for r in rows) / n,
        "cap_games": sum(1 for r in rows if r["reason"] == "cap"),
    }


def print_summary(s):
    """
    Prints a summary dictionary from summarize().

    Parameters:
    s (dict): The summary.

    Returns:
    None
    """
    print(f"\n{s['matchup']}  ({s['games']} games, seeds from {s['base_seed']})")
    print(f"  A wins {s['a_wins']}  B wins {s['b_wins']}  draws {s['draws']}")
    print(f"  A win rate {s['a_win_rate']:.3f}  95% CI [{s['a_ci_lo']:.3f}, {s['a_ci_hi']:.3f}]"
          f"   (as first {s['a_wins_as_first']}/{s['a_games_as_first']},"
          f" as second {s['a_wins_as_second']}/{s['a_games_as_second']})")
    print(f"  ms/move  A {s['a_ms_per_move']:.3f}  B {s['b_ms_per_move']:.3f}"
          f"   nodes/move  A {s['a_nodes_per_move']:.1f}  B {s['b_nodes_per_move']:.1f}")
    print(f"  avg plies {s['avg_plies']:.1f}   games ended by turn cap: {s['cap_games']}")


def append_rows(path, rows):
    """
    Appends dictionaries as rows of a csv file, writing the header if the file is new or empty.

    Parameters:
    path (str): The csv file.
    rows (list): Dictionaries that all have the same keys.

    Returns:
    None
    """
    is_new = not os.path.exists(path) or os.path.getsize(path) == 0
    with open(path, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        if is_new:
            writer.writeheader()
        writer.writerows(rows)


if __name__ == "__main__":
    # Quick demo: 10 games of Greedy (A) vs Random (B). Writes no files.
    from greedy import GreedyAgent, RandomAgent

    run_match(GreedyAgent(), RandomAgent(), n_games=10, out_dir=None)
