from alpha_zero_general import Coach
from alpha_zero_general import DotDict

from .game import OthelloGame
from .pytorch import OthelloNNet

args = DotDict(
    {
        "numIters": 100,
        "numEps": 100,  # Number of complete self-play games to simulate during a new iteration.
        "tempThreshold": 15,  #
        "maxlenOfQueue": 200000,  # Maximum number of game examples per iteration to train the neural networks
        "numMCTSSims": 40,  # Number of games moves for MCTS to simulate.
        "arenaCompare": 10,  # Number of games to play during arena play to determine if new net will be accepted.
        "updateThreshold": 0.60,  # During arena playoff, new neural net will be accepted if threshold or more of games are won.
        "cpuct": 1,
        "checkpoint": "./runs/1",
        "load_model": False,
        "load_folder_file": ("./runs/1", "model_00100"),
        "numItersForTrainExamplesHistory": 5,
        "nr_actors": 6,  # Number of self play episodes executed in parallel
    }
)

if __name__ == "__main__":
    game = OthelloGame(6)
    nnet = OthelloNNet(game)
    coach = Coach(game, nnet, args, pit_against_old_model=False)
    coach.learn()
