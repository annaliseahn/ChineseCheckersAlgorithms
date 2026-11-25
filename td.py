import argparse
import json
import random
import time

import numpy as np

from engine import BOARD_SIDE, create_board, get_valid_moves, make_move
from features import NUM_FEATURES, feature_vector

SCALE = (72, 6, 12)


def phi(board, player):
    f = feature_vector(board, player)
    x = []
    for i in range(NUM_FEATURES):
        x.append(f[i] / SCALE[i])
    return x


class TDAgent:
    def __init__(self, weight=None, gamma=1, epsilon=0.1, alpha=0.01, name="TD"):
        if weight is None:
            self.weight = [0, 0, 0]
        else:
            self.weight = weight
        self.gamma = gamma
        self.epsilon = epsilon
        self.alpha = alpha
        self.name = name
        self.last_nodes = 0
        self.episodes = 0
        self.train_seconds = 0.0

    def calc_value(self, board, player):
        x = phi(board, player)
        y = np.dot(self.weight, x)
        return y

    def choose_move(self, board, rng):
        """With probability epsilon pick a random action, otherwise the best-known one."""
        moves = get_valid_moves(board)
        self.last_nodes = len(moves)
        best_value = float("-inf")
        player = board.turn
        best_moves = []
        for move in moves:
            new_board = make_move(board, move)
            if new_board.game_over:
                if new_board.winner is None:
                    v = 0
                elif new_board.winner == player:
                    v = 1000000
                else:
                    v = -1000000
            else:
                v = self.calc_value(new_board, player)
            if v > best_value:
                best_value = v
                best_moves = [move]
            elif v == best_value:
                best_moves.append(move)
        return rng.choice(best_moves)

    def egreedy(self, board, rng):
        if rng.random() < self.epsilon:
            moves = get_valid_moves(board)
            return rng.choice(moves)
        else:
            return self.choose_move(board, rng)

    def update(self, board, player, target):
        x = phi(board, player)
        value = self.calc_value(board, player)
        error = target - value
        for i in range(NUM_FEATURES):
            self.weight[i] += self.alpha * error * x[i]

    def self_play_episode(self, rng):
        board = create_board()
        prev = [None, None]
        while True:
            q = board.turn
            move = self.egreedy(board, rng)
            board = make_move(board, move)
            if board.game_over:
                for p in range(2):
                    if prev[p] is not None:
                        if board.winner is None:
                            v = 0
                        elif board.winner == p:
                            v = 1
                        else:
                            v = -1
                        self.update(prev[p], p, v)
                return board.winner
            if prev[q] is not None:
                target = self.gamma * self.calc_value(board, q)
                self.update(prev[q], q, target)
            prev[q] = board

    def train(self, episodes, seed=0, log_every=500):
        rng = random.Random(seed)
        start = time.perf_counter()
        for n in range(1, episodes + 1):
            self.self_play_episode(rng)
            if log_every and n % log_every == 0:
                print(f"episode {n}: weight = {self.weight}")
        self.episodes += episodes
        self.train_seconds += time.perf_counter() - start

    def save(self, path="td_weights.json"):
        data = {
            "weight": self.weight,
            "episodes": self.episodes,
            "train_seconds": self.train_seconds,
            "gamma": self.gamma,
            "alpha": self.alpha,
            "epsilon": self.epsilon,
        }
        with open(path, "w") as f:
            json.dump(data, f, indent=1)

    @classmethod
    def load(cls, path="td_weights.json"):
        with open(path) as f:
            data = json.load(f)
            agent = cls(
                weight=data["weight"],
                gamma=data["gamma"],
                alpha=data["alpha"],
                epsilon=data["epsilon"],
            )
            agent.episodes = data["episodes"]
            agent.train_seconds = data["train_seconds"]
        return agent
