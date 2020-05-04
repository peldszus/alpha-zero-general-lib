import numpy as np

from .mcts import MCTS
from .utils import dotdict


class RandomPlayer:
    def __init__(self, game):
        self.game = game

    def play(self, board):
        a = np.random.randint(self.game.get_action_size())
        valids = self.game.get_valid_moves(board, 1)
        while valids[a] != 1:
            a = np.random.randint(self.game.get_action_size())
        return a


class HumanPlayer:
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
    def __init__(self, game):
        self.game = game

    def play(self, board):
        valids = self.game.get_valid_moves(board, 1)
        candidates = []
        for a in range(self.game.get_action_size()):
            if valids[a] == 0:
                continue
            nextBoard, _ = self.game.get_next_state(board, 1, a)
            score = self.game.getScore(nextBoard, 1)
            candidates += [(-score, a)]
        candidates.sort()
        return candidates[0][1]


class AlphaZeroPlayer:
    def __init__(
        self,
        game,
        nnwrapper_class,
        folder=None,
        filename=None,
        num_mcts_sims=50,
        cpuct=1.0,
    ):
        self.game = game
        self.net = nnwrapper_class(game)
        if folder and filename:
            self.net.load_checkpoint(folder, filename)
        self.args = dotdict({"numMCTSSims": num_mcts_sims, "cpuct": cpuct})
        self.mcts = MCTS(self.game, self.net, self.args)

    def play(self, board):
        return np.argmax(self.mcts.get_action_prob(board, temp=0))
