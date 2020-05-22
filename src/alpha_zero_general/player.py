"""
Generic player classes.
"""

from abc import ABC
from abc import abstractmethod

import numpy as np

from .mcts import MCTS
from .utils import DotDict


class Player(ABC):
    """The base player class."""

    @abstractmethod
    def play(self, board):
        """Returns the action of the player for the given board."""


class RandomPlayer(Player):
    """Selects a random valid action."""

    def __init__(self, game):
        self.game = game

    def play(self, board):
        """Returns the action of the player for the given board."""
        action = np.random.randint(self.game.get_action_size())
        valids = self.game.get_valid_moves(board, 1)
        while valids[action] != 1:
            action = np.random.randint(self.game.get_action_size())
        return action


class HumanPlayer(Player):
    """Selects an actions based on human input."""

    def __init__(self, game):
        self.game = game

    def play(self, board):
        """Returns the action of the player for the given board."""
        action_names = self.game.get_action_names()
        valids = self.game.get_valid_moves(board, 1)
        while True:
            input_action = input(self.game.get_action_prompt())
            if input_action not in action_names:
                print("Unknown action.")
                continue
            action = action_names[input_action]
            if valids[action] == 0:
                print("Invalid action.")
                continue
            else:
                break
        return action


class GreedyPlayer(Player):
    """
    Selects the action with the best immediate outcome according to
    a heuristic evaluation function game.get_score().
    """

    def __init__(self, game):
        self.game = game

    def play(self, board):
        """Returns the action of the player for the given board."""
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


class BareModelPlayer(Player):
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
        """Returns the action of the player for the given board."""
        valids = self.game.get_valid_moves(board, 1)
        pi, _ = self.net.predict(board)
        return np.argmax(pi * valids)


class AlphaZeroPlayer(Player):
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
        """Returns the action of the player for the given board."""
        return np.argmax(self.mcts.get_action_prob(board, temp=0))
