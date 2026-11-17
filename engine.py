import random

# Two-player Chinese Checkers engine (simplified board).
#
# Rules implemented
#   * A move is a single step to an adjacent empty cell, or a chain of one or more
#     hops. A hop jumps over an adjacent piece (either colour) onto the empty cell
#     directly behind it. A move is a pair (src, dst) of cell numbers; different hop
#     chains that end on the same cell count as the same move.
#   * A player wins when every cell of their goal triangle is occupied and at least
#     one of those pieces is theirs (so the opponent cannot block the goal forever).
#     Both players are checked after every move.
#   * Turn cap: after MAX_MOVES_PER_PLAYER moves each, the game ends and the player
#     with the lower total distance to their goal wins (equal distance = draw).
#   * Player 0 moves first. Player 0 starts at the top of print_board and heads for
#     the bottom; player 1 is the mirror image.

EMPTY = -1                  # value stored in board.owner for an empty cell
BOARD_SIDE = 3              # side length of each corner triangle (see Layout)
MAX_MOVES_PER_PLAYER = 150  # turn cap

# The six neighbour directions of a hex cell in cube coordinates (x, y, z).
DIRECTIONS = [(1, -1, 0), (1, 0, -1), (0, 1, -1), (-1, 1, 0), (-1, 0, 1), (0, -1, 1)]


###############################################################################
#                           Board Layout (never changes)                      #
###############################################################################

def hex_distance(a, b):
    """
    Distance between two cells given as cube coordinates.

    Parameters:
    a (tuple): (x, y, z) of the first cell.
    b (tuple): (x, y, z) of the second cell.

    Returns:
    int: Number of single steps between a and b on an empty board.
    """
    return max(abs(a[0] - b[0]), abs(a[1] - b[1]), abs(a[2] - b[2]))


class Layout:
    """
    Fixed geometry of the star-shaped board. It is built once per side length and
    shared by every Board of that size (see get_layout).

    Cells are cube coordinates (x, y, z) with x + y + z = 0, numbered 0..size-1. A
    cell is on the star if max(x, y, z) <= side or min(x, y, z) >= -side (two big
    triangles pointing opposite ways).
        side = 4 -> standard board: 121 cells, 10 pieces per player
        side = 3 -> simplified board: 73 cells, 6 pieces per player (default)
        side = 2 -> tiny board: 37 cells, 3 pieces per player (handy for debugging)

    Attributes:
    cells (list):      cells[i] is the (x, y, z) of cell i
    index (dict):      index[(x, y, z)] is the number of that cell
    size (int):        number of cells
    neighbors (list):  neighbors[i] lists the cells next to cell i
    hops (list):       hops[i] lists (over, land) pairs: a piece on cell i can jump
                       over cell `over` and land on cell `land`
    home (list):       home[p] lists the cells player p starts on
    goal (list):       goal[p] lists the cells player p has to fill (the other home)
    dist (list):       dist[p][i] is the distance from cell i to the tip of player
                       p's goal triangle
    num_pieces (int):  pieces per player
    """

    def __init__(self, side):
        self.side = side

        # Cells of the star (this loop order already gives sorted coordinates)
        self.cells = []
        for x in range(-2 * side, 2 * side + 1):
            for y in range(-2 * side, 2 * side + 1):
                z = -x - y
                if max(x, y, z) <= side or min(x, y, z) >= -side:
                    self.cells.append((x, y, z))
        self.size = len(self.cells)
        self.index = {}
        for i in range(self.size):
            self.index[self.cells[i]] = i

        # Neighbours and hops (index 0 is a real cell, so test "is not None")
        self.neighbors = []
        self.hops = []
        for (x, y, z) in self.cells:
            near = []
            jumps = []
            for dx, dy, dz in DIRECTIONS:
                over = self.index.get((x + dx, y + dy, z + dz))
                land = self.index.get((x + 2 * dx, y + 2 * dy, z + 2 * dz))
                if over is not None:
                    near.append(over)
                    if land is not None:
                        jumps.append((over, land))
            self.neighbors.append(near)
            self.hops.append(jumps)

        # Home / goal triangles: player 0 starts on the z <= -side-1 tip, player 1
        # on the opposite tip, and each heads for the other's starting tip.
        self.home = [[], []]
        for i in range(self.size):
            z = self.cells[i][2]
            if z <= -side - 1:
                self.home[0].append(i)
            elif z >= side + 1:
                self.home[1].append(i)
        self.goal = [self.home[1], self.home[0]]
        self.num_pieces = len(self.home[0])

        # Distance of every cell to the outermost cell (tip) of each player's goal
        tips = [(-side, -side, 2 * side), (side, side, -2 * side)]
        self.dist = []
        for p in (0, 1):
            self.dist.append([hex_distance(cell, tips[p]) for cell in self.cells])


_layouts = {}


def get_layout(side=BOARD_SIDE):
    """
    Returns the shared Layout for a side length, building it the first time.

    Parameters:
    side (int): Side length of the corner triangles (2, 3 or 4).

    Returns:
    Layout: The geometry of that board.
    """
    if side not in _layouts:
        _layouts[side] = Layout(side)
    return _layouts[side]


###############################################################################
#                                 Board (position)                            #
###############################################################################

class Board:
    """
    One game position. Boards are never changed after they are created:
    make_move returns a new Board.

    Attributes:
    layout (Layout):   the shared geometry
    owner (list):      owner[i] is EMPTY, 0 or 1 (who stands on cell i)
    pieces (list):     pieces[p] lists the cells where player p's pieces stand
    turn (int):        player to move, 0 or 1
    plies (int):       number of moves made so far by both players together
    game_over (bool):  True once the game has ended
    winner (int):      0 or 1 once somebody has won; None while playing and for a draw
    end_reason (str):  "goal" (a goal was filled), "cap" (turn cap) or "" while playing
    """

    def __init__(self, layout, owner, pieces, turn=0, plies=0):
        self.layout = layout
        self.owner = owner
        self.pieces = pieces
        self.turn = turn
        self.plies = plies
        self.game_over = False
        self.winner = None
        self.end_reason = ""


def create_board(side=BOARD_SIDE):
    """
    Creates the starting position: every player's pieces fill their home triangle.

    Parameters:
    side (int): Side length of the corner triangles (2, 3 or 4).

    Returns:
    Board: The starting board, player 0 to move.
    """
    layout = get_layout(side)
    owner = [EMPTY] * layout.size
    for p in (0, 1):
        for cell in layout.home[p]:
            owner[cell] = p
    pieces = [list(layout.home[0]), list(layout.home[1])]
    return Board(layout, owner, pieces)


###############################################################################
#                               Move Generation                               #
###############################################################################

def get_valid_moves(board):
    """
    Lists every move available to the player whose turn it is.

    Parameters:
    board (Board): The current board.

    Returns:
    list of tuple: (src, dst) cell numbers, in a fixed order. The list is empty if
    the game is over.
    """
    if board.game_over:
        return []

    layout = board.layout
    owner = board.owner
    moves = []
    for src in sorted(board.pieces[board.turn]):
        destinations = set()

        # 1) Single steps onto an adjacent empty cell
        for nb in layout.neighbors[src]:
            if owner[nb] == EMPTY:
                destinations.add(nb)

        # 2) Hop chains (depth-first search). The moving piece has left src, so it
        #    may not be hopped over and src itself counts as visited.
        visited = {src}
        stack = [src]
        while len(stack) != 0:
            curr = stack.pop()
            for over, land in layout.hops[curr]:
                if over != src and owner[over] != EMPTY and owner[land] == EMPTY and land not in visited:
                    visited.add(land)
                    destinations.add(land)
                    stack.append(land)

        for dst in sorted(destinations):
            moves.append((src, dst))
    return moves


def make_move(board, move):
    """
    Plays a move and returns the NEW board; the board passed in is not changed
    (like copying the board and calling drop_piece in connect four). The move should
    come from get_valid_moves(board): only the cheap checks below are made, so a
    destination that cannot be reached in one move is NOT detected.

    Parameters:
    board (Board): The current board.
    move (tuple): (src, dst) cell numbers.

    Returns:
    Board: The board after the move, with game_over / winner set if the game ended.

    Raises:
    ValueError: If src is not one of the mover's pieces or dst is not empty.
    """
    src, dst = move
    player = board.turn
    if board.owner[src] != player or board.owner[dst] != EMPTY:
        raise ValueError(f"Impossible move {move} for player {player}")
    child = Board(board.layout, board.owner[:], [board.pieces[0][:], board.pieces[1][:]],
                  1 - player, board.plies + 1)
    child.owner[src] = EMPTY
    child.owner[dst] = player
    my_pieces = child.pieces[player]
    my_pieces[my_pieces.index(src)] = dst

    # Did somebody fill their goal? Both players are checked, because a move can also
    # fill the OPPONENT's goal (a piece stepping back onto its own home triangle).
    for p in (0, 1):
        if winning_move(child, p):
            child.game_over = True
            child.winner = p
            child.end_reason = "goal"
            return child

    # Turn cap: the player closer to their goal wins, equal distance is a draw
    if child.plies >= 2 * MAX_MOVES_PER_PLAYER:
        child.game_over = True
        child.end_reason = "cap"
        d0 = total_distance(child, 0)
        d1 = total_distance(child, 1)
        if d0 < d1:
            child.winner = 0
        elif d1 < d0:
            child.winner = 1
    return child


###############################################################################
#                         Winning / Terminal / Distance                       #
###############################################################################

def winning_move(board, player):
    """
    Checks if the board is a winning position for the given player: every cell of
    their goal triangle is occupied and at least one of those pieces is theirs.

    Parameters:
    board (Board): The current board.
    player (int): 0 or 1.

    Returns:
    bool: True if the player has filled their goal, False otherwise.
    """
    has_own_piece = False
    for cell in board.layout.goal[player]:
        if board.owner[cell] == EMPTY:
            return False
        if board.owner[cell] == player:
            has_own_piece = True
    return has_own_piece


def is_terminal_node(board):
    """
    Checks if the game is over (a goal was filled or the turn cap was reached).

    Parameters:
    board (Board): The current board.

    Returns:
    bool: True if the game is over, False otherwise.
    """
    return board.game_over


def total_distance(board, player):
    """
    Sum over the player's pieces of the distance to the tip of their goal triangle.

    Parameters:
    board (Board): The current board.
    player (int): 0 or 1.

    Returns:
    int: Total distance (smaller is better).
    """
    dist = board.layout.dist[player]
    total = 0
    for cell in board.pieces[player]:
        total += dist[cell]
    return total


###############################################################################
#                                   Printing                                  #
###############################################################################

def print_board(board):
    """
    Prints the board as a star. X = player 0 (starts at the top), O = player 1
    (starts at the bottom), . = empty cell.

    Parameters:
    board (Board): The board to print.

    Returns:
    None
    """
    symbols = {EMPTY: ".", 0: "X", 1: "O"}
    rows = {}  # z -> list of (column, symbol)
    for i in range(board.layout.size):
        x, y, z = board.layout.cells[i]
        if z not in rows:
            rows[z] = []
        rows[z].append((2 * x + z, symbols[board.owner[i]]))

    left = min(col for row in rows.values() for col, _ in row)
    right = max(col for row in rows.values() for col, _ in row)
    for z in sorted(rows):
        line = [" "] * (right - left + 1)
        for col, symbol in rows[z]:
            line[col - left] = symbol
        print("".join(line).rstrip())


if __name__ == "__main__":
    random.seed(42)
    example_board = create_board()
    print("Debug: starting board (X moves first, toward the bottom)\n")
    print_board(example_board)
    print("\nNumber of valid moves for X:", len(get_valid_moves(example_board)))

    for _ in range(4):
        example_board = make_move(example_board, random.choice(get_valid_moves(example_board)))
    print("\nAfter 4 random moves:\n")
    print_board(example_board)
