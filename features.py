"""
Shared evaluation features for the agents.

Minimax, SA-tuned Minimax and TD learning all score a position with the same three
features, so they are written once, here:
    1) total distance to goal   (sum over the pieces; smaller is better)
    2) pieces in goal           (pieces already standing in the goal triangle)
    3) rearmost distance        (distance of the piece that is farthest from the goal)

feature_vector(board, player) is "player minus opponent" for each feature, so the
sign convention is the same for every agent, and

    score_position(board, player, weights) = weights . feature_vector(board, player)

Sensible weights are negative for the two distances and positive for pieces in goal.
SA tunes `weights`; TD learns them.
"""

from engine import total_distance

NUM_FEATURES = 3
DEFAULT_WEIGHTS = (-1.0, 2.0, -0.5)


def get_features(board, player):
    """
    Computes the three features for ONE player's pieces.

    Parameters:
    board (Board): The current board.
    player (int): 0 or 1.

    Returns:
    tuple: (total distance to goal, pieces in goal, distance of the rearmost piece)
    """
    dist = board.layout.dist[player]

    rearmost = 0
    for cell in board.pieces[player]:
        if dist[cell] > rearmost:
            rearmost = dist[cell]

    in_goal = 0
    for cell in board.layout.goal[player]:
        if board.owner[cell] == player:
            in_goal += 1

    return (total_distance(board, player), in_goal, rearmost)


def feature_vector(board, player):
    """
    Features of `player` minus the features of the opponent.

    Parameters:
    board (Board): The current board.
    player (int): 0 or 1, the player the numbers are from the point of view of.

    Returns:
    tuple: NUM_FEATURES numbers. The vector for the opponent is the negative of this.
    """
    mine = get_features(board, player)
    theirs = get_features(board, 1 - player)
    return tuple(mine[i] - theirs[i] for i in range(NUM_FEATURES))


def score_position(board, player, weights=DEFAULT_WEIGHTS):
    """
    Heuristic score of the board for `player` (higher is better for `player`).

    This only measures how far along the pieces are. It does not know that a game is
    over, so a search must test is_terminal_node(board) first and score wins and losses
    itself (like minimax in connect four does), before calling this at the depth limit.

    Parameters:
    board (Board): The current board.
    player (int): 0 or 1.
    weights (tuple): One weight per feature. Defaults to DEFAULT_WEIGHTS, so an agent
    with tuned (SA) or learned (TD) weights must pass them in.

    Returns:
    float: weights . feature_vector(board, player)
    """
    features = feature_vector(board, player)
    score = 0
    for i in range(NUM_FEATURES):
        score += weights[i] * features[i]
    return score
