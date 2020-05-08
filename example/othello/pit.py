from alpha_zero_general import AlphaZeroPlayer
from alpha_zero_general import Arena
from alpha_zero_general import BareModelPlayer
from alpha_zero_general import GreedyPlayer  # noqa
from alpha_zero_general import HumanPlayer  # noqa
from alpha_zero_general import RandomPlayer  # noqa

from .game import OthelloGame
from .nnet import OthelloNNet

"""
use this script to play any two agents against each other, or play manually with
any agent.
"""

game = OthelloGame(6)

folder, filename = "./temp/", "best.pth.tar"

# player1 = RandomPlayer(game)
# player1 = GreedyPlayer(game)
# player1 = HumanPlayer(game)
player1 = BareModelPlayer(game, OthelloNNet, folder, filename)
player2 = AlphaZeroPlayer(game, OthelloNNet, folder, filename)

arena = Arena(player1.play, player2.play, game, display=OthelloGame.display)

print(arena.play_games(10, verbose=True))
