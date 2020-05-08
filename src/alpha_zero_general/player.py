"""
Generic player classes.
"""

import numpy as np

from .mcts import MCTS
from .utils import DotDict


class RandomPlayer:
    """Selects a random valid action."""

    def __init__(self, game):
        self.game = game

    def play(self, board):
        action = np.random.randint(self.game.get_action_size())
        valids = self.game.get_valid_moves(board, 1)
        while valids[action] != 1:
            action = np.random.randint(self.game.get_action_size())
        return action


class HumanPlayer:
    """Selects an actions based on human input."""

    def __init__(self, game):
        self.game = game

    def play(self, board):
        # display(board)
        valid = self.game.get_valid_moves(board, 1)
        for i in range(len(valid)):
            if valid[i]:
                print(
                    "[", int(i / self.game.n), int(i % self.game.n), end="] "
                )
        while True:
            input_move = input()
            input_a = input_move.split(" ")
            if len(input_a) == 2:
                try:
                    x, y = [int(i) for i in input_a]
                    if (
                        (0 <= x)
                        and (x < self.game.n)
                        and (0 <= y)
                        and (y < self.game.n)
                    ) or ((x == self.game.n) and (y == 0)):
                        a = (
                            self.game.n * x + y
                            if x != -1
                            else self.game.n ** 2
                        )
                        if valid[a]:
                            break
                except ValueError:
                    # Input needs to be an integer
                    "Invalid integer"
            print("Invalid move")
        return a


class GreedyPlayer:
    """
    Selects the action with the best immediate outcome according to
    a heuristic evaluation function game.get_score().
    """

    def __init__(self, game):
        self.game = game

    def play(self, board):
        valids = self.game.get_valid_moves(board, 1)
        candidates = []
        for a in range(self.game.get_action_size()):
            if valids[a] == 0:
                continue
            next_board, _ = self.game.get_next_state(board, 1, a)
            score = self.game.get_score(next_board, 1)
            candidates += [(-score, a)]
        candidates.sort()
        return candidates[0][1]


class BareModelPlayer:
    """
    Selects the actions with the highest probability according to the model
    without simulating future steps using MCTS.
    """

    def __init__(
        self, game, nnet_class, folder=None, filename=None,
    ):
        self.game = game
        self.net = nnet_class(game)
        if folder and filename:
            self.net.load_checkpoint(folder, filename)

    def play(self, board):
        valids = self.game.get_valid_moves(board, 1)
        pi, _ = self.net.predict(board)
        return np.argmax(pi * valids)


class AlphaZeroPlayer:
    """
    Selects the actions with the best outcome according to the model when
    simulating future steps using MCTS.
    """

    def __init__(
        self,
        game,
        nnet_class,
        folder=None,
        filename=None,
        num_mcts_sims=50,
        cpuct=1.0,
    ):
        self.game = game
        self.net = nnet_class(game)
        if folder and filename:
            self.net.load_checkpoint(folder, filename)
        self.args = DotDict({"numMCTSSims": num_mcts_sims, "cpuct": cpuct})
        self.mcts = MCTS(self.game, self.net, self.args)

    def play(self, board):
        return np.argmax(self.mcts.get_action_prob(board, temp=0))
