import os
import random

import ray
from alpha_zero_general import Coach
from alpha_zero_general import DotDict
from alpha_zero_general.coach import ModelTrainer
from alpha_zero_general.coach import ReplayBuffer
from alpha_zero_general.coach import SelfPlay
from alpha_zero_general.coach import SharedStorage

from example.othello.game import OthelloGame
from example.othello.keras import OthelloNNet

args = DotDict(
    {
        "numIters": 2,
        "numEps": 2,  # Number of complete self-play games to simulate during a new iteration.
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


def test_shared_storage(local_ray):
    init_weights = [0, 0]
    init_revision = 1
    s = SharedStorage.remote(init_weights, revision=init_revision)
    assert ray.get(s.get_revision.remote()) == init_revision
    assert ray.get(s.get_weights.remote()) == (init_weights, init_revision)
    next_weights = [1, 1]
    next_revision = ray.get(s.set_weights.remote(next_weights, 0.5, 0.2))
    assert next_revision == init_revision + 1
    assert ray.get(s.get_weights.remote()) == (next_weights, next_revision)
    assert ray.get(s.get_infos.remote()) == {
        "trained_enough": False,
        "policy_loss": 0.5,
        "value_loss": 0.2,
    }
    assert ray.get(s.get_weights.remote(revision=next_revision + 1)) == (
        None,
        next_revision,
    )
    ray.get(s.set_info.remote("trained_enough", True))
    assert ray.get(s.trained_enough.remote()) is True


def test_replay_buffer(local_ray, tmpdir):
    def mock_game_examples(game=1, size=10):
        return [game] * size

    r = ReplayBuffer.remote(games_to_use=5, folder=tmpdir)
    assert ray.get(r.get_number_of_games_played.remote()) == 0
    game_1 = mock_game_examples(game=1)
    r.add_game_examples.remote(game_1)
    assert ray.get(r.get_number_of_games_played.remote()) == 1
    assert os.path.isfile(os.path.join(tmpdir, f"game_{1:08d}"))
    assert ray.get(ray.get(r.get_examples.remote())) == [game_1]
    for game in range(2, 7):
        r.add_game_examples.remote(mock_game_examples(game=game))
    assert ray.get(r.get_number_of_games_played.remote()) == 6
    games = ray.get(ray.get(r.get_examples.remote()))
    assert len(games) == 5
    assert games[0][0] == 2
    assert games[-1][0] == 6
    assert os.path.isfile(os.path.join(tmpdir, f"game_{6:08d}"))

    r = ReplayBuffer.remote(games_to_use=5, folder=tmpdir)
    assert ray.get(r.load.remote()) == 6
    games = ray.get(ray.get(r.get_examples.remote()))
    assert len(games) == 5
    assert games[0][0] == 2
    assert games[-1][0] == 6

    r = ReplayBuffer.remote(games_to_play=5, games_to_use=5, folder=tmpdir)
    assert ray.get(r.load.remote()) == 6
    assert ray.get(r.played_enough.remote()) is True

    r = ReplayBuffer.remote(games_to_play=10, games_to_use=5, folder=tmpdir)
    assert ray.get(r.load.remote()) == 6
    assert ray.get(r.played_enough.remote()) is False


def test_self_play(local_ray, tmpdir):
    game = OthelloGame(6)
    nnet = OthelloNNet(game)
    s = SharedStorage.remote(nnet.get_weights())
    r = ReplayBuffer.remote(games_to_play=1, games_to_use=1, folder=tmpdir)
    assert ray.get(r.get_number_of_games_played.remote()) == 0
    self_play = SelfPlay.remote(r, s, game, nnet.__class__, dict(args))
    ray.get(self_play.start.remote())
    assert ray.get(r.get_number_of_games_played.remote()) == 1
    assert ray.get(r.played_enough.remote()) is True
    games = ray.get(ray.get(r.get_examples.remote()))
    assert len(games) == 1
    examples = games[0]
    assert len(examples) > 2
    board, policy, winner = examples[0]
    assert isinstance(board, type(game.get_init_board()))
    assert len(policy) == game.get_action_size()
    assert all(0 <= value <= 1 for value in policy)
    assert winner in [1, -1]


def mock_example_data(game):
    board = game.get_init_board()
    pi = [random.random() for _ in range(game.get_action_size())]
    player = random.choice([1, -1])
    return [(b, p, player) for b, p in game.get_symmetries(board, pi)]


@ray.remote
class MockedReplayBuffer(ReplayBuffer.__ray_actor_class__):  # type: ignore
    """A replay buffer that behaves so that we'll go through all branches
    of ModelTrainer.start()."""

    played_enough_return_values = [False, False, False, True]

    def played_enough(self):
        """Returns preset values useful in this test."""
        return self.played_enough_return_values.pop(0)

    games_played_return_values = [0, 2, 4, 8]

    def get_number_of_games_played(self):
        """Returns preset values useful in this test."""
        return self.games_played_return_values.pop(0)


def test_model_trainer_loop(local_ray, tmpdir):
    game = OthelloGame(6)
    nnet = OthelloNNet(game)
    s = SharedStorage.remote(nnet.get_weights())
    assert ray.get(s.get_revision.remote()) == 0
    r = MockedReplayBuffer.remote(
        games_to_play=4, games_to_use=4, folder=tmpdir
    )
    r.add_game_examples.remote(mock_example_data(game))

    model_trainer = ModelTrainer.options(num_gpus=0).remote(
        r, s, game, nnet.__class__, dict(args), selfplay_training_ratio=1
    )
    ray.get(model_trainer.start.remote())
    assert ray.get(s.get_revision.remote()) > 0
    assert ray.get(s.trained_enough.remote()) is True


def test_model_trainer_pit_accept_model(capsys, local_ray, tmpdir):
    game = OthelloGame(6)
    nnet = OthelloNNet(game)
    s = SharedStorage.remote(nnet.get_weights())
    assert ray.get(s.get_revision.remote()) == 0
    r = ReplayBuffer.remote(games_to_play=2, games_to_use=2, folder=tmpdir)
    r.add_game_examples.remote(mock_example_data(game))
    # provoke model acceptance by tweaking updateThreshold to pass
    custom_args = dict(args, updateThreshold=-0.1)
    model_trainer = ModelTrainer.options(num_gpus=0).remote(
        r, s, game, nnet.__class__, custom_args, pit_against_old_model=True
    )
    ray.get(model_trainer.train.remote())
    assert ray.get(s.get_revision.remote()) == 1
    out, _err = capsys.readouterr()
    assert "PITTING AGAINST PREVIOUS VERSION" in out
    assert "ACCEPTING NEW MODEL" in out


def test_model_trainer_pit_reject_model(capsys, local_ray, tmpdir):
    game = OthelloGame(6)
    nnet = OthelloNNet(game)
    s = SharedStorage.remote(nnet.get_weights())
    assert ray.get(s.get_revision.remote()) == 0
    r = ReplayBuffer.remote(games_to_play=2, games_to_use=2, folder=tmpdir)
    r.add_game_examples.remote(mock_example_data(game))
    # provoke model rejection by tweaking updateThreshold to fail
    custom_args = dict(args, updateThreshold=1.1)
    model_trainer = ModelTrainer.options(num_gpus=0).remote(
        r, s, game, nnet.__class__, custom_args, pit_against_old_model=True
    )
    ray.get(model_trainer.train.remote())
    assert ray.get(s.get_revision.remote()) == 0
    out, _err = capsys.readouterr()
    assert "PITTING AGAINST PREVIOUS VERSION" in out
    assert "REJECTING NEW MODEL" in out


def test_coach(capsys, tmpdir):
    args.checkpoint = tmpdir
    game = OthelloGame(6)
    nnet = OthelloNNet(game)
    coach = Coach(game, nnet, args)
    coach.learn()
