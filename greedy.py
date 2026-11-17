"""
Baseline agents.

Agent interface (what harness.py expects; Minimax, SA-Minimax and TD should follow it):
    agent.name                       str, shown in the result files
    agent.new_game()                 optional, called before every game
    agent.choose_move(board, rng)    returns a (src, dst) move from get_valid_moves(board).
                                     rng is a seeded random.Random: use it for ALL
                                     randomness (tie-breaking) so games can be replayed.
    agent.last_nodes                 optional int, nodes expanded for the last move
"""

from engine import get_valid_moves


class GreedyAgent:
    """
    The fixed baseline opponent: looks one move ahead and plays the move that gets a
    piece closest to its goal (the biggest drop in total distance to goal). Equally
    good moves are chosen between uniformly at random, like a greedy() that breaks
    ties randomly instead of always taking the first best action.
    """

    name = "Greedy"
    last_nodes = 0

    def choose_move(self, board, rng):
        """
        Parameters:
        board (Board): The current board.
        rng (random.Random): Seeded random generator, used to break ties.

        Returns:
        tuple: The chosen (src, dst) move.
        """
        dist = board.layout.dist[board.turn]
        moves = get_valid_moves(board)

        best_gain = None
        best_moves = []
        for src, dst in moves:
            gain = dist[src] - dist[dst]  # how much closer to the goal the piece gets
            if best_gain is None or gain > best_gain:
                best_gain = gain
                best_moves = [(src, dst)]
            elif gain == best_gain:
                best_moves.append((src, dst))

        self.last_nodes = len(moves)
        return rng.choice(best_moves)


class RandomAgent:
    """Plays a uniformly random valid move. Only used as a sanity-check opponent."""

    name = "Random"
    last_nodes = 0

    def choose_move(self, board, rng):
        """
        Parameters:
        board (Board): The current board.
        rng (random.Random): Seeded random generator.

        Returns:
        tuple: A random (src, dst) move.
        """
        return rng.choice(get_valid_moves(board))
