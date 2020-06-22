import os
import tempfile

import ray
from alpha_zero_general import Coach
from alpha_zero_general import DotDict
from alpha_zero_general.coach import ReplayBuffer
from alpha_zero_general.coach import SharedStorage

from example.othello.game import OthelloGame
from example.othello.keras import OthelloNNet

args = DotDict(
    {
        "numIters": 2,
        "numEps": 10,  # Number of complete self-play games to simulate during a new iteration.
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


def test_shared_storage():
    init_weights = [0, 0]
    init_revision = 1
    ray.init()
    s = SharedStorage.remote(init_weights, revision=init_revision)
    assert ray.get(s.get_revision.remote()) == init_revision
    assert ray.get(s.get_weights.remote()) == (init_weights, init_revision)
    next_weights = [1, 1]
    next_revision = ray.get(s.set_weights.remote(next_weights, 0.5, 0.2))
    assert next_revision == init_revision + 1
    assert ray.get(s.get_weights.remote()) == (next_weights, next_revision)
    assert ray.get(s.get_infos.remote()) == {
        "policy_loss": 0.5,
        "value_loss": 0.2,
    }
    assert ray.get(s.get_weights.remote(revision=next_revision + 1)) == (
        None,
        next_revision,
    )
    ray.shutdown()


def test_replay_buffer():
    def mock_game_examples(game=1, size=10):
        return [game] * size

    with tempfile.TemporaryDirectory() as tmpdirname:
        ray.init()
        r = ReplayBuffer.remote(use_last_n_games=5, folder=tmpdirname)
        assert ray.get(r.get_number_of_games_played.remote()) == 0
        game_1 = mock_game_examples(game=1)
        r.add_game_examples.remote(game_1)
        assert ray.get(r.get_number_of_games_played.remote()) == 1
        assert os.path.isfile(os.path.join(tmpdirname, f"game_{1:08d}"))
        assert ray.get(ray.get(r.get_examples.remote())) == [game_1]
        for game in range(2, 7):
            r.add_game_examples.remote(mock_game_examples(game=game))
        assert ray.get(r.get_number_of_games_played.remote()) == 6
        games = ray.get(ray.get(r.get_examples.remote()))
        assert len(games) == 5
        assert games[0][0] == 2
        assert games[-1][0] == 6
        assert os.path.isfile(os.path.join(tmpdirname, f"game_{6:08d}"))
        ray.shutdown()

        ray.init()
        r = ReplayBuffer.remote(use_last_n_games=5, folder=tmpdirname)
        assert ray.get(r.load.remote()) == 6
        games = ray.get(ray.get(r.get_examples.remote()))
        assert len(games) == 5
        assert games[0][0] == 2
        assert games[-1][0] == 6
        ray.shutdown()


def test_coach_with_pit(capsys):
    with tempfile.TemporaryDirectory() as tmpdirname:
        args.checkpoint = tmpdirname
        game = OthelloGame(6)
        nnet = OthelloNNet(game)
        coach = Coach(game, nnet, args, pit_against_old_model=True)
        coach.learn()
        out, _err = capsys.readouterr()
        assert "PITTING AGAINST PREVIOUS VERSION" in out
        assert "ACCEPTING NEW MODEL" in out or "REJECTING NEW MODEL" in out
