"""
Minimax Agents.

Agent interface (what harness.py expects; Minimax, SA-Minimax and TD should follow it):
    agent.name                       str, shown in the result files
    agent.new_game()                 optional, called before every game
    agent.minimax(board, depth, maximizing_player)  returns (score, move)
"""

from engine import get_valid_moves, is_terminal_node, total_distance, make_move

class MinimaxAgent:
    """
    The Minimax agent: looks ahead a fixed number of moves and chooses 
    the move that maximizes its chances of winning (or minimizes the opponent's 
    chances). It uses a simple evaluation function based
    on the total distance of pieces to their goal.
    """

    name = "Minimax"
    last_nodes = 0

    def __init__(self, depth=3):
        """
        Initializes the Minimax agent with a specified search depth.

        Parameters:
        depth (int): The depth to search in the game tree.
        """
        self.depth = depth

    def new_game(self):
        """
        Resets the agent's state for a new game.
        """
        self.last_nodes = 0

    def minimax(self, board, depth, maximizing_player, alpha=float('-inf'), 
                beta=float('inf'), root_player=None):
        """
        Performs minimax with alpha-beta pruning to select the best possible 
        move to make.

        Parameters:
        board (Board): The current board.
        depth (int): The depth to search in the game tree.
        maximizing_player (bool): True if the current player is the maximizing player.
        alpha (float): The best value that the maximizer currently can guarantee at that level or above.
        beta (float): The best value that the minimizer currently can guarantee at that level or above.
        root_player: The
        """

        if root_player is None:
            root_player = board.turn

        # Increase reward for landing on winning node
        if is_terminal_node(board):
            if board.winner == root_player:
                return 1000, None
            elif board.winner == 1 - root_player:
                return -1000, None
            else:
                return 0, None

        # If end of depth is reached, final policy eval score is higher if current
        # player's total distance from goal is less than opponent's
        if depth == 0:
            score = (total_distance(board, 1 - root_player) 
                     - total_distance(board, root_player))
            return score, None
        
        self.last_nodes += 1
        valid_moves = get_valid_moves(board)

        if not valid_moves:
            score = (
                total_distance(board, 1 - root_player)
                - total_distance(board, root_player)
            )
            return score, None

        best_move = None

        if maximizing_player:
            best_score = float('-inf')

            for move in valid_moves:
                # Simulate current board to not affect actual game board
                new_board = make_move(board, move)

                score, _ = self.minimax(
                    new_board, depth - 1, False,
                    alpha, beta, root_player
                )

                if score > best_score:
                    best_score = score
                    best_move = move

                alpha = max(alpha, best_score)

                if beta <= alpha:
                    break

            return best_score, best_move

        else:
            best_score = float('inf')

            for move in valid_moves:
                # Simulate current board to not affect actual game board
                new_board = make_move(board, move)

                score, _ = self.minimax(
                    new_board, depth - 1, True,
                    alpha, beta, root_player
                )

                if score < best_score:
                    best_score = score
                    best_move = move

                beta = min(beta, best_score)

                if beta <= alpha:
                    break

            return best_score, best_move
        
    # Gets best move as calculated by minimax with AB-pruning
    def choose_move(self, board, rng):
        """
        Selects move using minimax algorithm

        Parameters:
        board (Board): The current board.
        rng (random.Random): Seeded random generator.

        Returns:
        tuple: The chosen (src, dst) move.
        """
        self.last_nodes = 0
        _, best_move = self.minimax(board, self.depth, True)
        return best_move