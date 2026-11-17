"""
Tests for the engine, features, agents and harness.   Run:  python test_engine.py

The main safety net is test_moves_match_reference: reference_moves() below writes the
move rules again from scratch (on x, y, z coordinates, without the neighbour/hop tables
of engine.py) and every position of many games must give exactly the same moves.
"""

import csv
import os
import random
import tempfile
import unittest

import engine
from engine import (EMPTY, DIRECTIONS, Board, get_layout, create_board, get_valid_moves,
                    make_move, winning_move, is_terminal_node, total_distance)
from features import get_features, feature_vector, score_position
from greedy import GreedyAgent, RandomAgent
from harness import play_game, run_match, append_rows, wilson_ci


###############################################################################
#                                   Helpers                                   #
###############################################################################

def make_board(side, cells0, cells1, turn=0, plies=0):
    """
    Builds a board with player 0's pieces on cells0 and player 1's on cells1.

    Parameters:
    side (int): Board size.
    cells0 (list): (x, y, z) cells of player 0's pieces.
    cells1 (list): (x, y, z) cells of player 1's pieces.
    turn (int): Player to move.
    plies (int): Moves made so far.

    Returns:
    Board: The board.
    """
    layout = get_layout(side)
    owner = [EMPTY] * layout.size
    pieces = [[], []]
    for player, cells in ((0, cells0), (1, cells1)):
        for cell in cells:
            owner[layout.index[cell]] = player
            pieces[player].append(layout.index[cell])
    return Board(layout, owner, pieces, turn, plies)


def cell_number(board, cell):
    return board.layout.index[cell]


def reference_moves(board):
    """
    The move rules written again from scratch, to cross-check get_valid_moves.

    Parameters:
    board (Board): A board where the game is not over.

    Returns:
    tuple: (set of (src, dst) moves, True if some destination needs a chain of 2+ hops)
    """
    layout = board.layout
    occupied = set()
    for i in range(layout.size):
        if board.owner[i] != EMPTY:
            occupied.add(layout.cells[i])

    moves = set()
    has_chain = False
    for src in board.pieces[board.turn]:
        start = layout.cells[src]
        others = occupied - {start}  # the moving piece has left its cell

        single_hops = set()
        for dx, dy, dz in DIRECTIONS:
            step = (start[0] + dx, start[1] + dy, start[2] + dz)
            land = (start[0] + 2 * dx, start[1] + 2 * dy, start[2] + 2 * dz)
            if step in layout.index and step not in others:
                moves.add((src, layout.index[step]))
            if step in others and land in layout.index and land not in others:
                single_hops.add(land)

        reachable = set()
        stack = [start]
        while len(stack) != 0:
            x, y, z = stack.pop()
            for dx, dy, dz in DIRECTIONS:
                over = (x + dx, y + dy, z + dz)
                land = (x + 2 * dx, y + 2 * dy, z + 2 * dz)
                if (over in others and land in layout.index and land not in others
                        and land != start and land not in reachable):
                    reachable.add(land)
                    stack.append(land)

        if reachable - single_hops:
            has_chain = True
        for land in reachable:
            moves.add((src, layout.index[land]))
    return moves, has_chain


def playout(side, agent0, agent1, seed):
    """
    Plays one game and returns every board of it, from the start to the end.

    Parameters:
    side (int): Board size.
    agent0 (object): Agent playing player 0.
    agent1 (object): Agent playing player 1.
    seed (int): Seed of the random generator given to the agents.

    Returns:
    list of Board: All boards in order.
    """
    rng = random.Random(seed)
    agents = (agent0, agent1)
    board = create_board(side)
    boards = [board]
    while not is_terminal_node(board):
        board = make_move(board, agents[board.turn].choose_move(board, rng))
        boards.append(board)
    return boards


def mirrored(board):
    """
    The same position turned by 180 degrees with the two players swapped. The rules are
    symmetric, so it must have the mirror images of the same moves.

    Parameters:
    board (Board): The board to mirror.

    Returns:
    tuple: (mirrored board, function that maps a cell number to its mirror cell number)
    """
    layout = board.layout

    def mirror_cell(i):
        x, y, z = layout.cells[i]
        return layout.index[(-x, -y, -z)]

    owner = [EMPTY] * layout.size
    for i in range(layout.size):
        if board.owner[i] != EMPTY:
            owner[mirror_cell(i)] = 1 - board.owner[i]
    pieces = [[mirror_cell(i) for i in board.pieces[1 - p]] for p in (0, 1)]
    return Board(layout, owner, pieces, 1 - board.turn, board.plies), mirror_cell


def snapshot(board):
    return (board.owner[:], [board.pieces[0][:], board.pieces[1][:]], board.turn,
            board.plies, board.game_over, board.winner, board.end_reason)


###############################################################################
#                                    Layout                                   #
###############################################################################

class TestLayout(unittest.TestCase):
    def test_sizes(self):
        for side, cells, pieces in [(2, 37, 3), (3, 73, 6), (4, 121, 10)]:
            layout = get_layout(side)
            self.assertEqual(layout.size, cells)
            self.assertEqual(layout.num_pieces, pieces)

    def test_symmetry(self):
        for side in (2, 3, 4):
            layout = get_layout(side)
            for i in range(layout.size):
                x, y, z = layout.cells[i]
                self.assertIn((-z, -x, -y), layout.index)  # turned by 60 degrees
                self.assertIn((-x, -y, -z), layout.index)  # turned by 180 degrees
                for j in layout.neighbors[i]:
                    self.assertIn(i, layout.neighbors[j])

    def test_homes_and_goals(self):
        for side in (2, 3, 4):
            layout = get_layout(side)
            self.assertEqual(set(layout.home[0]) & set(layout.home[1]), set())
            self.assertEqual(layout.goal[0], layout.home[1])
            self.assertEqual(layout.goal[1], layout.home[0])
            for p in (0, 1):
                # a triangle with rows of 1, 2, ..., side cells: distances 0, 1, 1, 2, 2, 2, ...
                distances = sorted(layout.dist[p][i] for i in layout.goal[p])
                self.assertEqual(distances, [row for row in range(side) for _ in range(row + 1)])
                # no other set of cells is as close to the tip as the goal triangle
                everything = sorted(layout.dist[p])
                self.assertEqual(everything[:layout.num_pieces], distances)
                self.assertLess(everything[layout.num_pieces - 1], everything[layout.num_pieces])


###############################################################################
#                                Move generation                              #
###############################################################################

class TestMoves(unittest.TestCase):
    def test_opening_move_counts(self):
        # counted by hand: steps of the front row + hops of the second row over the front row
        for side, expected in [(2, 6), (3, 10), (4, 14)]:
            self.assertEqual(len(get_valid_moves(create_board(side))), expected)

    def test_step_and_single_hop(self):
        board = make_board(3, [(0, 0, 0)], [(1, -1, 0)])
        destinations = {board.layout.cells[dst] for src, dst in get_valid_moves(board)}
        self.assertIn((2, -2, 0), destinations)      # hops over the blocker
        self.assertNotIn((1, -1, 0), destinations)   # cannot land on a piece
        self.assertEqual(len(destinations), 5 + 1)   # 5 free neighbours + 1 hop

    def test_hop_chain(self):
        board = make_board(3, [(0, 0, 0)], [(1, -1, 0), (2, -1, -1)])
        destinations = {board.layout.cells[dst] for src, dst in get_valid_moves(board)}
        self.assertIn((2, -2, 0), destinations)      # first hop
        self.assertIn((2, 0, -2), destinations)      # second hop, over (2, -1, -1)

    def test_own_pieces_can_be_hopped(self):
        board = make_board(3, [(0, 0, 0), (1, -1, 0)], [(-3, 3, 0)])
        destinations = {board.layout.cells[dst] for src, dst in get_valid_moves(board)
                        if board.layout.cells[src] == (0, 0, 0)}
        self.assertIn((2, -2, 0), destinations)

    def test_no_hop_over_empty_cells(self):
        board = make_board(3, [(0, 0, 0)], [(3, -3, 0)])
        destinations = {board.layout.cells[dst] for src, dst in get_valid_moves(board)}
        self.assertEqual(len(destinations), 6)       # only the 6 steps

    def test_boards_are_not_changed(self):
        board = create_board(3)
        before = snapshot(board)
        moves = get_valid_moves(board)
        self.assertEqual(snapshot(board), before)
        child = make_move(board, moves[0])
        self.assertEqual(snapshot(board), before)    # the parent is untouched
        self.assertEqual(child.turn, 1)
        self.assertEqual(child.plies, 1)
        self.assertEqual(child.owner[moves[0][0]], EMPTY)
        self.assertEqual(child.owner[moves[0][1]], 0)

    def test_impossible_moves_are_rejected(self):
        board = create_board(3)
        src, dst = get_valid_moves(board)[0]
        other = board.pieces[1][0]
        with self.assertRaises(ValueError):
            make_move(board, (src, other))  # the destination is occupied
        with self.assertRaises(ValueError):
            make_move(board, (other, dst))  # the piece belongs to the other player

    def test_moves_match_reference(self):
        agent_pairs = [(RandomAgent(), RandomAgent()), (GreedyAgent(), RandomAgent()),
                       (GreedyAgent(), GreedyAgent())]
        chain_positions = 0
        for side in (2, 3, 4):
            for seed in range(3):
                for agent0, agent1 in agent_pairs:
                    for board in playout(side, agent0, agent1, seed):
                        moves = get_valid_moves(board)
                        if is_terminal_node(board):
                            self.assertEqual(moves, [])
                            continue
                        expected, has_chain = reference_moves(board)
                        self.assertEqual(len(moves), len(set(moves)))  # no duplicates
                        self.assertEqual(set(moves), expected)
                        chain_positions += has_chain
        self.assertGreater(chain_positions, 100)  # the test really did cover hop chains

    def test_mirror_symmetry(self):
        for side in (2, 3, 4):
            for board in playout(side, RandomAgent(), RandomAgent(), 5)[::7]:
                if is_terminal_node(board):
                    continue
                other, mirror_cell = mirrored(board)
                expected = {(mirror_cell(src), mirror_cell(dst)) for src, dst in get_valid_moves(board)}
                self.assertEqual(set(get_valid_moves(other)), expected)


###############################################################################
#                                  End of game                                #
###############################################################################

class TestGameEnd(unittest.TestCase):
    def test_nobody_has_won_at_the_start(self):
        board = create_board(3)
        self.assertFalse(winning_move(board, 0))  # the goal is full, but of the opponent's pieces
        self.assertFalse(winning_move(board, 1))
        self.assertFalse(is_terminal_node(board))

    def test_filling_the_goal_wins(self):
        board = make_board(2, [(-2, -2, 4), (-2, -1, 3), (-1, -1, 2)], [(0, 1, -1), (0, -1, 1), (1, -1, 0)])
        child = make_move(board, (cell_number(board, (-1, -1, 2)), cell_number(board, (-1, -2, 3))))
        self.assertTrue(is_terminal_node(child))
        self.assertEqual(child.winner, 0)
        self.assertEqual(child.end_reason, "goal")
        self.assertEqual(get_valid_moves(child), [])

    def test_a_blocker_cannot_stop_the_win(self):
        # player 1 keeps a piece on (-1, -2, 3) in player 0's goal; the other two goal cells
        # are player 0's, so the goal counts as filled
        board = make_board(2, [(-2, -2, 4), (-1, -1, 2)], [(-1, -2, 3), (0, 1, -1), (1, -1, 0)])
        child = make_move(board, (cell_number(board, (-1, -1, 2)), cell_number(board, (-2, -1, 3))))
        self.assertTrue(is_terminal_node(child))
        self.assertEqual(child.winner, 0)

    def test_filling_the_opponents_goal_ends_the_game(self):
        # player 1 holds 2 of the 3 cells of its goal (the top tip); player 0 steps back
        # onto the third one, so that goal is full and contains player 1's pieces
        board = make_board(2, [(0, 2, -2), (0, 0, 0), (-1, 0, 1)], [(2, 2, -4), (2, 1, -3), (0, 1, -1)])
        child = make_move(board, (cell_number(board, (0, 2, -2)), cell_number(board, (1, 2, -3))))
        self.assertTrue(is_terminal_node(child))
        self.assertEqual(child.winner, 1)
        self.assertEqual(child.end_reason, "goal")

    def test_turn_cap_goes_to_the_closer_player(self):
        board = make_board(2, [(-2, -2, 4), (-2, -1, 3), (-1, -1, 2)], [(0, 0, 0), (0, 1, -1), (1, 0, -1)],
                           turn=1, plies=2 * engine.MAX_MOVES_PER_PLAYER - 1)
        child = make_move(board, get_valid_moves(board)[0])
        self.assertTrue(is_terminal_node(child))
        self.assertEqual(child.end_reason, "cap")
        self.assertEqual(child.winner, 0)
        self.assertEqual(get_valid_moves(child), [])

    def test_turn_cap_with_equal_distance_is_a_draw(self):
        board = make_board(2, [(1, -1, 0)], [(-1, 1, 0)], turn=1, plies=2 * engine.MAX_MOVES_PER_PLAYER - 1)
        child = make_move(board, (cell_number(board, (-1, 1, 0)), cell_number(board, (0, 0, 0))))
        self.assertEqual(total_distance(child, 0), total_distance(child, 1))
        self.assertTrue(is_terminal_node(child))
        self.assertEqual(child.end_reason, "cap")
        self.assertIsNone(child.winner)

    def test_games_end_correctly(self):
        for seed in range(6):
            boards = playout(3, RandomAgent(), RandomAgent(), seed)
            self.assertLessEqual(len(boards) - 1, 2 * engine.MAX_MOVES_PER_PLAYER)
            for before, after in zip(boards, boards[1:]):
                self.assertFalse(is_terminal_node(before))
                self.assertEqual(after.plies, before.plies + 1)
                self.assertEqual(after.turn, 1 - before.turn)
            last = boards[-1]
            if last.end_reason == "goal":
                self.assertTrue(winning_move(last, last.winner))
            else:
                self.assertEqual(last.plies, 2 * engine.MAX_MOVES_PER_PLAYER)


###############################################################################
#                       Features, agents and the harness                      #
###############################################################################

class SpyAgent(GreedyAgent):
    """Greedy that remembers which player number it was asked to move for."""
    name = "Spy"

    def __init__(self):
        self.turns = set()

    def choose_move(self, board, rng):
        self.turns.add(board.turn)
        return super().choose_move(board, rng)


class CheatingAgent:
    """
    Teleports one of its own pieces to the highest-numbered empty cell. make_move's cheap
    checks accept that (own piece, empty cell), so only the harness's check can catch it.
    """
    name = "Cheater"

    def choose_move(self, board, rng):
        empty = [i for i in range(board.layout.size) if board.owner[i] == EMPTY]
        return (board.pieces[board.turn][0], empty[-1])


class GlobalRandomAgent:
    """Picks moves with the global random module, like random.choice in the connect four minimax."""
    name = "GlobalRandom"

    def choose_move(self, board, rng):
        return random.choice(get_valid_moves(board))


class ListAgent(GreedyAgent):
    """Returns its move as a list instead of a tuple."""
    name = "ListAgent"

    def choose_move(self, board, rng):
        return list(super().choose_move(board, rng))


class CheatsInThirdGame(GreedyAgent):
    """Plays normally for two games, then returns an illegal move."""
    name = "CheatsInThirdGame"

    def __init__(self):
        self.games = 0

    def new_game(self):
        self.games += 1

    def choose_move(self, board, rng):
        if self.games >= 3:
            return (board.pieces[board.turn][0], board.layout.size - 1)
        return super().choose_move(board, rng)


def read_csv(path):
    with open(path, newline="") as f:
        return list(csv.DictReader(f))


class TestFeatures(unittest.TestCase):
    def test_start_is_even(self):
        board = create_board(3)
        self.assertEqual(feature_vector(board, 0), (0, 0, 0))
        self.assertEqual(score_position(board, 0), 0)

    def test_features_on_a_small_board(self):
        # distances are to the tip of each player's goal: (-2, -2, 4) for player 0, (2, 2, -4) for player 1
        board = make_board(2, [(-2, -2, 4), (-2, -1, 3), (0, 0, 0)], [(2, 2, -4), (0, 1, -1), (-1, -2, 3)])
        self.assertEqual(get_features(board, 0), (0 + 1 + 4, 2, 4))   # (-1, -2, 3) is in the goal but not theirs
        self.assertEqual(get_features(board, 1), (0 + 3 + 7, 1, 7))
        self.assertEqual(feature_vector(board, 0), (-5, 1, -3))
        self.assertEqual(score_position(board, 0), 8.5)               # -1 * -5 + 2 * 1 + -0.5 * -3

    def test_first_feature_is_total_distance(self):
        for board in playout(3, GreedyAgent(), RandomAgent(), 2)[::9]:
            for p in (0, 1):
                self.assertEqual(get_features(board, p)[0], total_distance(board, p))

    def test_opponent_sees_the_negative(self):
        for board in playout(3, GreedyAgent(), RandomAgent(), 1)[::5]:
            self.assertEqual(feature_vector(board, 0), tuple(-v for v in feature_vector(board, 1)))

    def test_progress_raises_the_score(self):
        board = create_board(3)
        move = GreedyAgent().choose_move(board, random.Random(0))
        self.assertGreater(score_position(make_move(board, move), 0), score_position(board, 0))


class TestAgentsAndHarness(unittest.TestCase):
    def test_games_are_reproducible(self):
        first = play_game(GreedyAgent(), RandomAgent(), True, 7)
        second = play_game(GreedyAgent(), RandomAgent(), True, 7)
        for key in ("winner", "plies", "a_final_dist", "b_final_dist"):
            self.assertEqual(first[key], second[key])

    def test_greedy_only_plays_best_moves_and_breaks_ties_randomly(self):
        board = create_board(3)
        dist = board.layout.dist[0]
        gains = [dist[src] - dist[dst] for src, dst in get_valid_moves(board)]
        chosen = {GreedyAgent().choose_move(board, random.Random(seed)) for seed in range(40)}
        for src, dst in chosen:
            self.assertEqual(dist[src] - dist[dst], max(gains))
        self.assertGreater(len(chosen), 1)  # several moves are equally good at the start

    def test_greedy_beats_random(self):
        wins = sum(play_game(GreedyAgent(), RandomAgent(), g % 2 == 0, g)["winner"] == "A" for g in range(10))
        self.assertGreaterEqual(wins, 9)

    def test_seats(self):
        for a_first in (True, False):
            spy = SpyAgent()
            result = play_game(spy, GreedyAgent(), a_first, 3)
            self.assertEqual(spy.turns, {0} if a_first else {1})
            self.assertEqual(result["first_mover"], "A" if a_first else "B")
            self.assertEqual(result["a_moves"] + result["b_moves"], result["plies"])

    def test_illegal_moves_are_rejected(self):
        with self.assertRaises(ValueError):
            play_game(CheatingAgent(), GreedyAgent(), True, 0)

    def test_moves_given_as_lists_are_accepted(self):
        result = play_game(ListAgent(), GreedyAgent(), True, 2)
        self.assertIn(result["winner"], ("A", "B", "draw"))

    def test_logged_moves_replay_to_the_same_result(self):
        result = play_game(GreedyAgent(), RandomAgent(), True, 11)
        board = create_board(3)
        for text in result["move_list"].split():
            move = tuple(int(v) for v in text.split("-"))
            self.assertIn(move, get_valid_moves(board))
            board = make_move(board, move)
        self.assertTrue(is_terminal_node(board))
        self.assertEqual(board.plies, result["plies"])
        self.assertEqual(board.winner, {"A": 0, "B": 1, "draw": None}[result["winner"]])  # A moved first

    def test_agents_using_the_global_random_module_replay_exactly(self):
        random.seed(1)
        first = play_game(GlobalRandomAgent(), GreedyAgent(), True, 4)
        random.seed(2)  # whatever the global generator was doing before, the game replays the same
        second = play_game(GlobalRandomAgent(), GreedyAgent(), True, 4)
        self.assertEqual(first["move_list"], second["move_list"])

    def test_the_callers_global_random_state_is_left_alone(self):
        random.seed(3)
        before = random.getstate()
        play_game(GlobalRandomAgent(), GreedyAgent(), True, 4)
        self.assertEqual(random.getstate(), before)

    def test_wilson_interval(self):
        low, high = wilson_ci(50, 100)
        self.assertAlmostEqual(low, 0.4038, places=3)
        self.assertAlmostEqual(high, 0.5962, places=3)
        self.assertEqual(wilson_ci(0, 0), (0.0, 0.0))
        self.assertEqual(wilson_ci(100, 100)[1], 1.0)  # winning every game: the upper end is exactly 1
        self.assertEqual(wilson_ci(0, 10)[0], 0.0)


class TestMatchFiles(unittest.TestCase):
    def test_games_and_summary_are_written_and_appended(self):
        with tempfile.TemporaryDirectory() as folder:
            summary = run_match(GreedyAgent(), RandomAgent(), n_games=4, out_dir=folder, verbose=False)
            games = read_csv(os.path.join(folder, "games.csv"))
            self.assertEqual([g["game"] for g in games], ["0", "1", "2", "3"])
            self.assertEqual([g["first_mover"] for g in games], ["A", "B", "A", "B"])
            rows = read_csv(os.path.join(folder, "summary.csv"))
            self.assertEqual(len(rows), 1)
            self.assertEqual(int(rows[0]["a_wins"]), summary["a_wins"])
            self.assertEqual(int(rows[0]["a_games_as_first"]), 2)
            self.assertEqual(int(rows[0]["a_games_as_second"]), 2)

            run_match(GreedyAgent(), RandomAgent(), n_games=2, out_dir=folder, verbose=False)
            self.assertEqual(len(read_csv(os.path.join(folder, "games.csv"))), 6)    # appended
            self.assertEqual(len(read_csv(os.path.join(folder, "summary.csv"))), 2)

    def test_summary_adds_up(self):
        summary = run_match(GreedyAgent(), RandomAgent(), n_games=6, out_dir=None, verbose=False)
        self.assertEqual(summary["a_wins"] + summary["b_wins"] + summary["draws"], 6)
        self.assertEqual(summary["a_wins_as_first"] + summary["a_wins_as_second"], summary["a_wins"])
        self.assertAlmostEqual(summary["a_win_rate"], summary["a_wins"] / 6)
        self.assertLessEqual(summary["a_ci_lo"], summary["a_win_rate"])
        self.assertGreaterEqual(summary["a_ci_hi"], summary["a_win_rate"])

    def test_nothing_is_written_without_out_dir(self):
        with tempfile.TemporaryDirectory() as folder:
            start_folder = os.getcwd()
            os.chdir(folder)
            try:
                summary = run_match(GreedyAgent(), RandomAgent(), n_games=2, out_dir=None, verbose=False)
                self.assertEqual(os.listdir(folder), [])
            finally:
                os.chdir(start_folder)
        self.assertEqual(summary["games"], 2)

    def test_finished_games_survive_a_crash(self):
        with tempfile.TemporaryDirectory() as folder:
            with self.assertRaises(ValueError):
                run_match(CheatsInThirdGame(), GreedyAgent(), n_games=6, out_dir=folder, verbose=False)
            self.assertEqual(len(read_csv(os.path.join(folder, "games.csv"))), 2)

    def test_header_is_written_into_an_empty_file(self):
        with tempfile.TemporaryDirectory() as folder:
            path = os.path.join(folder, "empty.csv")
            open(path, "w").close()
            append_rows(path, [{"a": 1, "b": 2}])
            self.assertEqual(read_csv(path), [{"a": "1", "b": "2"}])


if __name__ == "__main__":
    unittest.main()
