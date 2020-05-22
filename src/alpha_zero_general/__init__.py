from .arena import Arena
from .coach import Coach
from .game import Game
from .mcts import MCTS
from .neuralnet import NeuralNet
from .player import AlphaZeroPlayer
from .player import BareModelPlayer
from .player import GreedyPlayer
from .player import HumanPlayer
from .player import Player
from .player import RandomPlayer
from .utils import DotDict

from .league import League  # isort:skip (League must be imported after Player)

__all__ = [
    "AlphaZeroPlayer",
    "BareModelPlayer",
    "Arena",
    "Coach",
    "DotDict",
    "Game",
    "GreedyPlayer",
    "League",
    "HumanPlayer",
    "Player",
    "MCTS",
    "NeuralNet",
    "RandomPlayer",
]

__version__ = "0.1.0"
