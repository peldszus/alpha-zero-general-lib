from .arena import Arena
from .coach import Coach
from .game import Game
from .mcts import MCTS
from .neuralnet import NeuralNet
from .player import AlphaZeroPlayer
from .player import BareModelPlayer
from .player import GreedyPlayer
from .player import HumanPlayer
from .player import RandomPlayer
from .utils import DotDict

__all__ = [
    "AlphaZeroPlayer",
    "BareModelPlayer",
    "Arena",
    "Coach",
    "DotDict",
    "Game",
    "GreedyPlayer",
    "HumanPlayer",
    "MCTS",
    "NeuralNet",
    "RandomPlayer",
]

__version__ = "0.1.0"
