from alpha_zero_general import AlphaZeroPlayer
from alpha_zero_general import Arena
from alpha_zero_general import GreedyPlayer
from alpha_zero_general import HumanPlayer  # noqa
from alpha_zero_general import RandomPlayer  # noqa

from NNet import NNetWrapper
from OthelloGame import OthelloGame

"""
use this script to play any two agents against each other, or play manually with
any agent.
"""

game = OthelloGame(6)

player1 = GreedyPlayer(game)

folder, filename = "./temp/", "best.pth.tar"
player2 = AlphaZeroPlayer(game, NNetWrapper, folder, filename)

arena = Arena(player1.play, player2.play, game, display=OthelloGame.display)

print(arena.play_games(2, verbose=True))
