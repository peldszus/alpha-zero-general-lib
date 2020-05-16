from collections import Counter

from alpha_zero_general import Coach
from alpha_zero_general import DotDict

from example.othello.game import OthelloGame
from example.othello.keras import OthelloNNet

args = DotDict(
    {
        "numIters": 2,
        "numEps": 1,  # Number of complete self-play games to simulate during a new iteration.
        "tempThreshold": 15,  #
        "updateThreshold": 0.6,  # During arena playoff, new neural net will be accepted if threshold or more of games are won.
        "maxlenOfQueue": 10,  # Number of game examples to train the neural networks.
        "numMCTSSims": 2,  # Number of games moves for MCTS to simulate.
        "arenaCompare": 2,  # Number of games to play during arena play to determine if new net will be accepted.
        "cpuct": 1,
        "checkpoint": "/tmp/alpha_zero_general/",
        "load_model": False,
        "load_folder_file": ("/tmp/alpha_zero_general/", "best.pth.tar"),
        "numItersForTrainExamplesHistory": 20,
        "nr_actors": 2,  # Number of self play episodes executed in parallel
    }
)


def test_coach(capsys):
    game = OthelloGame(6)
    nnet = OthelloNNet(game)
    coach = Coach(game, nnet, args)
    coach.learn()
    out, _err = capsys.readouterr()
    counter = Counter(out.splitlines())
    assert counter["------ITER 1------"] == 1
    assert counter["------ITER 2------"] == 1
    assert counter["PITTING AGAINST PREVIOUS VERSION"] == 2
    assert counter["ACCEPTING NEW MODEL"] + counter["REJECTING NEW MODEL"] == 2
