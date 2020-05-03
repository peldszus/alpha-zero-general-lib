import numpy as np
from alpha_zero_general import MCTS
from alpha_zero_general import Arena
from alpha_zero_general import GreedyPlayer
from alpha_zero_general import HumanPlayer
from alpha_zero_general import RandomPlayer
from alpha_zero_general import dotdict

from NNet import NNetWrapper
from OthelloGame import OthelloGame

"""
use this script to play any two agents against each other, or play manually with
any agent.
"""

mini_othello = True  # Play in 6x6 instead of the normal 8x8.
human_vs_cpu = False

if mini_othello:
    g = OthelloGame(6)
else:
    g = OthelloGame(8)

# all players
rp = RandomPlayer(g).play
gp = GreedyPlayer(g).play
hp = HumanPlayer(g).play


# nnet players
n1 = NNetWrapper(g)
if mini_othello:
    n1.load_checkpoint("./temp/", "checkpoint_1.pth.tar")
else:
    n1.load_checkpoint(
        "./pretrained_models/othello/pytorch/",
        "8x8_100checkpoints_best.pth.tar",
    )
args1 = dotdict({"numMCTSSims": 50, "cpuct": 1.0})
mcts1 = MCTS(g, n1, args1)
n1p = lambda x: np.argmax(mcts1.getActionProb(x, temp=0))

if human_vs_cpu:
    player2 = hp
else:
    n2 = NNetWrapper(g)
    n2.load_checkpoint("./temp/", "checkpoint_1.pth.tar")
    args2 = dotdict({"numMCTSSims": 50, "cpuct": 1.0})
    mcts2 = MCTS(g, n2, args2)
    n2p = lambda x: np.argmax(mcts2.getActionProb(x, temp=0))

    player2 = n2p  # Player 2 is neural network if it's cpu vs cpu.

arena = Arena(n1p, player2, g, display=OthelloGame.display)

print(arena.playGames(2, verbose=True))
