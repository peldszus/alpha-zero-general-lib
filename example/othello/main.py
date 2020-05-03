from alpha_zero_general import Coach
from alpha_zero_general import dotdict

from NNet import NNetWrapper
from OthelloGame import OthelloGame

args = dotdict(
    {
        "numIters": 100,
        "numEps": 10,  # Number of complete self-play games to simulate during a new iteration.
        "tempThreshold": 15,  #
        "updateThreshold": 0.6,  # During arena playoff, new neural net will be accepted if threshold or more of games are won.
        "maxlenOfQueue": 200000,  # Number of game examples to train the neural networks.
        "numMCTSSims": 5,  # Number of games moves for MCTS to simulate.
        "arenaCompare": 10,  # Number of games to play during arena play to determine if new net will be accepted.
        "cpuct": 1,
        "checkpoint": "./temp/",
        "load_model": False,
        "load_folder_file": ("/dev/models/8x100x50", "best.pth.tar"),
        "numItersForTrainExamplesHistory": 20,
    }
)

if __name__ == "__main__":
    g = OthelloGame(6)
    nnet = NNetWrapper(g)

    if args.load_model:
        nnet.load_checkpoint(
            args.load_folder_file[0], args.load_folder_file[1]
        )

    c = Coach(g, nnet, args)
    if args.load_model:
        print("Load trainExamples from file")
        c.loadTrainExamples()
    c.learn()
