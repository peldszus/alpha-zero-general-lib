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

from .arena import Arena
from .mcts import MCTS
from .utils import DotDict

# *** Async feature ***
# TODO:
# - Enable continuing training (model)
# - Add option to not save model early models
# - Fix get_examples() takes too long over ray
# - Adjust tests
#


@ray.remote
class SelfPlayActor:
    def __init__(self, replay_buffer, weight_storage, game, nnet_class, args):
        self.replay_buffer = replay_buffer
        self.weight_storage = weight_storage
        self.game = game
        self.args = DotDict(args)
        self.nnet = nnet_class(self.game)
        self.mcts = None
        self.model_revision = -1

    def start(self):
        """Start the main self play loop for this actor."""
        while True:
            weights, self.model_revision = ray.get(
                self.weight_storage.get_weights.remote(self.model_revision)
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
class WeightStorage:
    """
    Class which is run in a dedicated thread to store the network weights.
    """

    def __init__(self, weights, revision=0):
        self.weights = weights
        self.revision = revision
        print(f"Initialized with weights at revision {self.revision}.")

    def get_weights(self, revision=0):
        """
        Return the weights and the current revision, if the given revision
        is older than the current one. Otherwise return None-type weights,
        to save bandwidth.
        """
        if revision < self.revision:
            print("Providing newest weights.")
            return self.weights, self.revision
        else:
            return None, self.revision

    def get_revision(self):
        """Return the current revision of the model weights."""
        return self.revision

    def set_weights(self, weights):
        """Set the next revision of the model weights."""
        self.weights = weights
        self.revision += 1
        return self.revision


@ray.remote
class ReplayBuffer:
    """
    Class which is run in a dedicated thread to store played games and generate batch.
    """

    def __init__(self, use_last_n_games=2000, folder=None):
        self.history = deque([], maxlen=use_last_n_games)
        self.use_last_n_games = use_last_n_games
        self.games_played = 0
        self.folder = folder

    def get_examples(self):
        """Returns the examples from the history of played games."""
        examples = np.array(
            [example for game in self.history for example in game]
        )
        # TODO: Returning the examples actually takes quite long, maybe because ray needs
        # to pickle and unpickle up to 2000 games (500mb)...?
        # > Done fetching 522472 examples in 368.90213775634766 seconds (1416 e/s).
        return examples

    def add_game_examples(self, examples):
        """Add examples of a recent game."""
        self.history.append(examples)
        self.games_played += 1
        self.save(examples, self.games_played)

    def get_number_of_games_played(self):
        """Return the number of games played."""
        return self.games_played

    def load(self):
        """Loads game examples from folder and return the total number of
        games played."""
        print(f"Loading played games from {self.folder}...")
        filenames = glob.glob(os.path.join(self.folder, "game_*"))
        self.games_played = len(filenames)
        # TODO: parse `games_played` out of filename
        for filename in sorted(filenames)[-self.use_last_n_games :]:
            print(f"Loading from {filename}...")
            with open(filename, "rb") as f:
                self.history.append(Unpickler(f).load())
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


@ray.remote(num_gpus=1)
class ModelTrainer:
    def __init__(
        self,
        replay_buffer,
        weight_storage,
        game,
        nnet_class,
        args,
        pit_against_old_model=False,
    ):
        self.replay_buffer = replay_buffer
        self.weight_storage = weight_storage
        self.game = game
        self.args = DotDict(args)
        self.pit_against_old_model = pit_against_old_model
        self.nnet_class = nnet_class
        self.nnet = None
        self.pnet = None
        self.model_revision = -1

    def start(self):
        # get initial weights
        self.nnet = self.nnet_class(self.game)
        weights, self.model_revision = ray.get(
            self.weight_storage.get_weights.remote(self.model_revision)
        )
        self.nnet.set_weights(weights)

        # wait for the first played games
        while (
            ray.get(self.replay_buffer.get_number_of_games_played.remote()) < 1
        ):
            time.sleep(0.2)

        # train loop
        while True:
            old_weights = self.nnet.get_weights()
            print(f"Training next model revision {self.model_revision}...")
            print("Fetching last examples...")
            t1 = time.time()
            train_examples = ray.get(self.replay_buffer.get_examples.remote())
            t = time.time() - t1
            print(
                f"Done fetching {len(train_examples)} examples in {t} seconds ({int(len(train_examples)/t)} e/s)."
            )
            shuffle(train_examples)
            self.nnet.train(train_examples)
            weights = self.nnet.get_weights()
            self.model_revision = ray.get(
                self.weight_storage.set_weights.remote(weights)
            )
            self.nnet.save_checkpoint(
                folder=self.args.checkpoint,
                filename=f"model_{self.model_revision:05d}",
            )

            if self.pit_against_old_model:
                if self.wins_against_old_model(old_weights):
                    self.nnet.save_checkpoint(
                        folder=self.args.checkpoint, filename="model_best"
                    )
                else:
                    self.nnet.set_weights(old_weights)

    def wins_against_old_model(self, old_weights):
        """Returns True if the current model won pitting against the last one."""
        # TODO: Like training, this should be done on GPU if possible.
        print("PITTING AGAINST PREVIOUS VERSION")
        old_net = self.nnet_class(self.game)
        old_net.set_weights(old_weights)
        pmcts = MCTS(self.game, old_net, self.args)
        nmcts = MCTS(self.game, self.nnet, self.args)
        arena = Arena(
            lambda x: np.argmax(pmcts.get_action_prob(x, temp=0)),
            lambda x: np.argmax(nmcts.get_action_prob(x, temp=0)),
            self.game,
        )
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
    This class executes the self-play + learning. It uses the functions defined
    in Game and NeuralNet. args are specified in main.py.
    """

    def __init__(self, game, nnet, args, pit_against_old_model=False):
        self.game = game
        self.nnet = nnet
        self.nnet_class = nnet.__class__
        self.args = args
        self.pit_against_old_model = pit_against_old_model

    def learn(self):
        """
        Performs numIters iterations with numEps episodes of self-play in each
        iteration. After every iteration, it retrains neural network with
        examples in train_examples (which has a maximum length of maxlenofQueue).
        It then pits the new neural network against the old one and accepts it
        only if it wins >= updateThreshold fraction of games.
        """
        games_to_play = self.args.numEps * self.args.numIters
        games_to_use = (
            self.args.numEps * self.args.numItersForTrainExamplesHistory
        )

        if self.args.load_model:
            self.nnet.load_checkpoint(
                folder=self.args.load_folder_file[0],
                filename=self.args.load_folder_file[1],
            )

        # initialize components
        ray.init(ignore_reinit_error=True)

        # TODO: be able to continue training
        weight_storage = WeightStorage.remote(self.nnet.get_weights())
        del self.nnet

        replay_buffer = ReplayBuffer.remote(
            use_last_n_games=games_to_use, folder=self.args.checkpoint,
        )
        ray.get(replay_buffer.load.remote())

        self_play_actor_pool = [
            SelfPlayActor.remote(
                replay_buffer,
                weight_storage,
                deepcopy(self.game),
                self.nnet_class,
                dict(self.args),
            )
            for _ in range(self.args.nr_actors)
        ]
        # model_trainer = ModelTrainer.options(
        #     num_gpus=1 if self.nnet.model.args.cuda else 0
        # )
        model_trainer = ModelTrainer.remote(
            replay_buffer,
            weight_storage,
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
        while (
            ray.get(replay_buffer.get_number_of_games_played.remote())
            < games_to_play
        ):
            # TODO add progress bar here
            time.sleep(0.5)

        # close
        ray.shutdown()

        #         with tqdm(
        #             desc="Self play episodes", total=self.args.numEps
        #         ) as bar:
        #             for train_examples in actor_pool.map_unordered(
        #                 lambda a, v: a.execute_episode.remote(v),
        #                 range(1, self.args.numEps + 1),
        #             ):
        #                 iteration_train_examples += train_examples
        #                 bar.update(1)
