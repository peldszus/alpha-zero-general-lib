"""
An asynchronous implementation of the general alpha-zero algorithm.
"""

import glob
import os
import time
from collections import deque
from copy import deepcopy
from pickle import Pickler
from pickle import Unpickler
from random import shuffle

import numpy as np
import ray
from tqdm import tqdm

from .arena import Arena
from .mcts import MCTS
from .player import AlphaZeroPlayer
from .utils import DotDict
from .utils import parse_game_filename
from .utils import parse_model_filename


@ray.remote
class SelfPlay:
    """Actor to execute self play."""

    def __init__(self, replay_buffer, shared_storage, game, nnet_class, args):
        self.replay_buffer = replay_buffer
        self.shared_storage = shared_storage
        self.game = game
        self.args = DotDict(args)
        self.nnet = nnet_class(self.game)
        self.mcts = None
        self.model_revision = -1

    def start(self):
        """Start the main self play loop for this actor and close when done."""
        while not ray.get(self.replay_buffer.played_enough.remote()):
            weights, self.model_revision = ray.get(
                self.shared_storage.get_weights.remote(self.model_revision)
            )
            if weights:
                self.nnet.set_weights(weights)
            examples = self.execute_episode()
            self.replay_buffer.add_game_examples.remote(examples)

    def execute_episode(self):
        """
        This function executes one episode of self-play, starting with player 1.
        As the game is played, each turn is added as a training example to
        train_examples. The game is played till the game ends. After the game
        ends, the outcome of the game is used to assign values to each example
        in train_examples.

        It uses a temp=1 if episode_step < tempThreshold, and thereafter
        uses temp=0.

        Returns:
            train_examples: a list of examples of the form
                            (canonical_board, current_player, pi,v)
                            pi is the MCTS informed policy vector, v is +1 if
                            the player eventually won the game, else -1.
        """  # TODO: incorrect description of return
        self.mcts = MCTS(self.game, self.nnet, self.args)
        train_examples = []
        board = self.game.get_init_board()
        current_player = 1
        episode_step = 0

        while True:
            episode_step += 1
            canonical_board = self.game.get_canonical_form(
                board, current_player
            )
            temp = int(episode_step < self.args.tempThreshold)

            pi = self.mcts.get_action_prob(canonical_board, temp=temp)
            sym = self.game.get_symmetries(canonical_board, pi)
            for b, p in sym:
                train_examples.append([b, current_player, p, None])

            action = np.random.choice(len(pi), p=pi)
            board, current_player = self.game.get_next_state(
                board, current_player, action
            )

            r = self.game.get_game_ended(board, current_player)

            if r != 0:
                return [
                    (x[0], x[2], r * ((-1) ** (x[1] != current_player)))
                    for x in train_examples
                ]


@ray.remote
class SharedStorage:
    """Actor to store and provide the most recent network weights and infos."""

    def __init__(self, weights, revision=0):
        self.weights = weights
        self.revision = revision
        self.infos = {
            "trained_enough": False,
            "policy_loss": None,
            "value_loss": None,
        }
        print(f"Initialized with weights at revision {self.revision}.")

    def get_weights(self, revision=0):
        """
        Return the weights and the current revision, if the given revision
        is older than the current one. Otherwise return None-type weights.
        """
        if revision < self.revision:
            return self.weights, self.revision
        else:
            return None, self.revision

    def get_revision(self):
        """Return the current revision of the model weights."""
        return self.revision

    def set_weights(self, weights, policy_loss=None, value_loss=None):
        """Set the next revision of the model weights."""
        self.weights = weights
        if policy_loss:
            self.set_info("policy_loss", policy_loss)
        if value_loss:
            self.set_info("value_loss", value_loss)
        self.revision += 1
        return self.revision

    def get_infos(self):
        """Returns the stored information dictionary."""
        return self.infos

    def set_info(self, key, value):
        """Set or update a value in the information dictionary."""
        self.infos[key] = value

    def trained_enough(self):
        """Returns true if the last iteration of training is done."""
        return self.infos["trained_enough"]


@ray.remote
class ReplayBuffer:
    """Actor to store played games and provide the latest examples for training."""

    def __init__(self, games_to_play=1000, games_to_use=500, folder=None):
        self.history = deque([], maxlen=games_to_use)
        self.games_to_play = games_to_play
        self.games_to_use = games_to_use
        self.games_played = 0
        self.folder = folder

    def get_examples(self):
        """Returns a list of (ray object ids of) examples from the history of played games."""
        return list(self.history)

    def add_game_examples(self, examples):
        """Add examples of a recent game."""
        self.history.append(ray.put(examples))
        self.games_played += 1
        self.save(examples, self.games_played)

    def get_number_of_games_played(self):
        """Return the number of games played."""
        return self.games_played

    def played_enough(self):
        """Returns true if all the number of requested games has been played."""
        return self.games_played >= self.games_to_play

    def load(self):
        """Loads game examples from folder and return the total number of
        games played."""
        print(f"Loading played games from {self.folder}...")
        filenames = sorted(glob.glob(os.path.join(self.folder, "game_*")))
        self.games_played = 0
        if filenames:
            self.games_played = parse_game_filename(
                os.path.basename(filenames[-1])
            )
        for filename in filenames[-self.games_to_use :]:
            print(f"Loading from {filename}...")
            with open(filename, "rb") as f:
                examples = Unpickler(f).load()
                self.history.append(ray.put(examples))
        print("Done loading.")
        return self.games_played

    def save(self, examples, number_of_games_played):
        """Saves game examples to folder."""
        if not os.path.exists(self.folder):
            os.makedirs(self.folder)
        filename = os.path.join(
            self.folder, f"game_{number_of_games_played:08d}"
        )
        with open(filename, "wb+") as f:
            Pickler(f).dump(examples)


@ray.remote
class ModelTrainer:
    """
    Actor to train the model.

    The `selfplay_training_ratio` controls how often the model is supposed
    to be updated in comparison to played games. A ration of 2.0 means to
    update every second played game, a ratio of 0.5 means update twice for
    each played game.

    If `pit_against_old_model` is True, the trainer will test a new revision
    of the neural network against the old one and only accept it if it wins
    >= updateThreshold fraction of games.
    """

    def __init__(
        self,
        replay_buffer,
        shared_storage,
        game,
        nnet_class,
        args,
        selfplay_training_ratio=2.0,
        pit_against_old_model=False,
        save_model_from_revision_n_on=0,
    ):
        self.replay_buffer = replay_buffer
        self.shared_storage = shared_storage
        self.game = game
        self.args = DotDict(args)
        self.pit_against_old_model = pit_against_old_model
        self.nnet_class = nnet_class
        self.nnet = None
        self.model_revision = -1
        self.selfplay_training_ratio = selfplay_training_ratio
        self.save_model_from_revision_n_on = save_model_from_revision_n_on
        # get initial weights
        self.nnet = self.nnet_class(self.game)
        weights, self.model_revision = ray.get(
            self.shared_storage.get_weights.remote(self.model_revision)
        )
        self.nnet.set_weights(weights)

    def start(self):
        """Start the main training loop and close when done."""
        # train loop, we train as long as we play
        while not ray.get(self.replay_buffer.played_enough.remote()):

            # wait with training according to selfplay / training ratio
            games_played, model_revision = ray.get(
                [
                    self.replay_buffer.get_number_of_games_played.remote(),
                    self.shared_storage.get_revision.remote(),
                ]
            )
            if (
                games_played / max(1, model_revision)
                <= self.selfplay_training_ratio
            ):
                time.sleep(0.5)
                continue

            self.train()

        # close
        ray.get(self.shared_storage.set_info.remote("trained_enough", True))

    def train(self):
        """Trains the model one more iteration."""
        old_weights = self.nnet.get_weights()
        game_object_ids = ray.get(self.replay_buffer.get_examples.remote())
        games = ray.get(game_object_ids)
        train_examples = [example for game in games for example in game]
        shuffle(train_examples)
        policy_loss, value_loss = self.nnet.train(train_examples)
        weights = self.nnet.get_weights()

        if self.pit_against_old_model and not self.wins_against_old_model(
            old_weights
        ):
            # reject the model
            self.nnet.set_weights(old_weights)
            return
        else:
            self.model_revision = ray.get(
                self.shared_storage.set_weights.remote(
                    weights, policy_loss, value_loss
                )
            )
            if self.model_revision >= self.save_model_from_revision_n_on:
                self.nnet.save_checkpoint(
                    folder=self.args.checkpoint,
                    filename=f"model_{self.model_revision:05d}",
                )

    def wins_against_old_model(self, old_weights):
        """Returns True if the current model won pitting against the last one."""
        # TODO: Like training, this should be done on GPU if possible.
        print("PITTING AGAINST PREVIOUS VERSION")
        old_net = self.nnet_class(self.game)
        old_net.set_weights(old_weights)
        kwargs = dict(
            num_mcts_sims=self.args.numMCTSSims, cpuct=self.args.cpuct
        )
        prev_model_player = AlphaZeroPlayer(self.game, old_net, **kwargs)
        new_model_player = AlphaZeroPlayer(self.game, self.nnet, **kwargs)
        arena = Arena(prev_model_player, new_model_player, self.game)
        pwins, nwins, draws = arena.play_games(self.args.arenaCompare)
        print("NEW/PREV WINS : %d / %d ; DRAWS : %d" % (nwins, pwins, draws))
        if (
            pwins + nwins == 0
            or float(nwins) / (pwins + nwins) < self.args.updateThreshold
        ):
            print("REJECTING NEW MODEL")
            return False
        else:
            print("ACCEPTING NEW MODEL")
            return True


class Coach:
    """
    This class executes the alpha zero like learning scheme.

    Upon start it spawns
    - a shared storage, which holds current model weights and infos,
    - a replay buffer, which holds recent games played,
    - a pool of actors for selfplay, which produce new games with the current
      model weights,
    - and the model trainer, which updates the model with the recent games.

    It uses the functions defined in Game and NeuralNet.
    Args are specified in main.py.
    """

    def __init__(self, game, nnet, args, pit_against_old_model=False):
        self.game = game
        self.nnet = nnet
        self.nnet_class = nnet.__class__
        self.args = args
        self.pit_against_old_model = pit_against_old_model
        self.request_gpu = self.nnet.request_gpu()

    def learn(self):
        """Start the learning algorithm."""
        games_to_play = self.args.numEps * self.args.numIters
        games_to_use = (
            self.args.numEps * self.args.numItersForTrainExamplesHistory
        )

        revision = 0
        if self.args.load_model:
            print(
                f"Loading initial model from checkpoint {self.args.load_folder_file}..."
            )
            revision = parse_model_filename(self.args.load_folder_file[1])
            self.nnet.load_checkpoint(
                folder=self.args.load_folder_file[0],
                filename=self.args.load_folder_file[1],
            )

        # initialize components
        ray.init(ignore_reinit_error=True)

        shared_storage = SharedStorage.remote(
            self.nnet.get_weights(), revision=revision
        )
        del self.nnet

        replay_buffer = ReplayBuffer.remote(
            games_to_play=games_to_play,
            games_to_use=games_to_use,
            folder=self.args.checkpoint,
        )
        ray.get(replay_buffer.load.remote())

        self_play_actor_pool = [
            SelfPlay.remote(
                replay_buffer,
                shared_storage,
                deepcopy(self.game),
                self.nnet_class,
                dict(self.args),
            )
            for _ in range(self.args.nr_actors)
        ]
        model_trainer = ModelTrainer.options(
            num_gpus=1 if self.request_gpu else 0
        ).remote(
            replay_buffer,
            shared_storage,
            deepcopy(self.game),
            self.nnet_class,
            dict(self.args),
            pit_against_old_model=self.pit_against_old_model,
        )

        # start self play and model trainer
        for actor in self_play_actor_pool:
            actor.start.remote()
        model_trainer.start.remote()

        # wait until all games are played
        t = tqdm(
            desc="Self played games",
            total=games_to_play,
            bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}{postfix}]",
        )
        while not all(
            ray.get(
                [
                    replay_buffer.played_enough.remote(),
                    shared_storage.trained_enough.remote(),
                ]
            )
        ):
            games_played, revision, infos = ray.get(
                [
                    replay_buffer.get_number_of_games_played.remote(),
                    shared_storage.get_revision.remote(),
                    shared_storage.get_infos.remote(),
                ]
            )
            t.set_postfix(
                model=revision,
                pi_loss=infos["policy_loss"],
                v_loss=infos["value_loss"],
            )
            t.update(games_played - t.n)
            time.sleep(0.5)
        t.close()

        # close
        ray.shutdown()
