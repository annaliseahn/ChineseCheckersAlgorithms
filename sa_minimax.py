from minimax import MinimaxAgent
from engine import get_valid_moves, make_move
import random
import math

class tuned_minimax_agent(MinimaxAgent):
    """
    A tuned version of the Minimax agent with a depth of 3.
    """

    name = "SA-Minimax"
    last_nodes = 0

    def __init__(self, depth=3, initial_temp=1.0, temperature=1.0, cooling_rate=0.99, 
                 min_temp=0.1, rng=None):
        super().__init__(depth)
        self.initial_temp = temperature
        self.temperature = temperature
        self.cooling_rate = cooling_rate
        self.min_temp = min_temp
        self.rng = rng if rng is not None else random.Random()

    def new_game(self):
        super().new_game()
        self.temperature = self.initial_temp
    
    def choose_move(self, board, rng):
        """
        Gets values of move-actions using minimax, then has a chance to explore
        rather than exploit based on simulated annealing temperature.

        Parameters:
        board (Board): The current board.
        rng (random.Random): Seeded random generator.

        Returns:
        tuple: The chosen (src, dst) move.
        """
        
        last_nodes = 0
        valid_moves = get_valid_moves(board)

        if not valid_moves:
            return None 
        
        # Get the best move using the Minimax agent
        best_move = super().choose_move(board, rng)

        # Get the evaluation of the best move
        new_board = make_move(board, best_move)
        best_eval, _ = self.minimax(new_board, self.depth - 1, False, 
                                    float('-inf'), float('inf'), root_player=board.turn)
        
        # Calculate probabilities for each move using stable softmax
        move_probs = []

        for move in valid_moves:
            new_board = make_move(board, move)
            eval, _ = self.minimax(
                new_board,
                self.depth - 1,
                False,
                float('-inf'),
                float('inf'),
                root_player=board.turn
            )

            # Get probability of move being selected; modulated by temperature
            prob = math.exp((eval - best_eval) / max(self.temperature, 1e-8)) # Makes sure temp is positive
            move_probs.append(prob)

        # Normalize probabilities
        total_prob = sum(move_probs)
        move_probs = [prob / total_prob for prob in move_probs]

        # Choose a move based on the probabilities
        chosen_move = rng.choices(valid_moves, weights=move_probs, k=1)[0]

        # Cool down the temperature
        self.temperature = max(self.min_temp, self.temperature * self.cooling_rate)

        return chosen_move
    
    

        

    
    