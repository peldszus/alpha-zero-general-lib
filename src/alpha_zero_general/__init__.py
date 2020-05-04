from .arena import Arena
from .coach import Coach
from .game import Game
from .mcts import MCTS
from .neuralnet import NeuralNet
from .player import GreedyPlayer
from .player import HumanPlayer
from .player import NeuralNetPlayer
from .player import RandomPlayer
from .utils import dotdict

__all__ = [
    "Arena",
    "Coach",
    "dotdict",
    "Game",
    "GreedyPlayer",
    "HumanPlayer",
    "MCTS",
    "NeuralNet",
    "NeuralNetPlayer",
    "RandomPlayer",
]

__version__ = "0.1.0"
