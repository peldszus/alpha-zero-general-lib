import os
import sys
from collections import deque
from pickle import Pickler
from pickle import Unpickler
from random import shuffle

import numpy as np
from tqdm import tqdm

from .arena import Arena
from .mcts import MCTS


class Coach:
    """
    This class executes the self-play + learning. It uses the functions defined
    in Game and NeuralNet. args are specified in main.py.
    """

    def __init__(self, game, nnet, args):
        self.game = game
        self.nnet = nnet
        self.pnet = self.nnet.__class__(self.game)  # the competitor network
        self.args = args
        self.mcts = MCTS(self.game, self.nnet, self.args)
        # history of examples from args.numItersForTrainExamplesHistory latest iterations
        self.train_examples_history = []
        # can be overriden in load_train_examples()
        self.skip_first_self_play = False

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
        """
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

    def learn(self):
        """
        Performs numIters iterations with numEps episodes of self-play in each
        iteration. After every iteration, it retrains neural network with
        examples in train_examples (which has a maximum length of maxlenofQueue).
        It then pits the new neural network against the old one and accepts it
        only if it wins >= updateThreshold fraction of games.
        """

        for i in range(1, self.args.numIters + 1):
            # bookkeeping
            print("------ITER " + str(i) + "------")
            # examples of the iteration
            if not self.skip_first_self_play or i > 1:
                iteration_train_examples = deque(
                    [], maxlen=self.args.maxlenOfQueue
                )

                for _ in tqdm(range(self.args.numEps), desc="Self Play"):
                    # reset search tree
                    self.mcts = MCTS(self.game, self.nnet, self.args)
                    iteration_train_examples += self.execute_episode()

                # save the iteration examples to the history
                self.train_examples_history.append(iteration_train_examples)

            if (
                len(self.train_examples_history)
                > self.args.numItersForTrainExamplesHistory
            ):
                print(
                    "len(train_examples_history) =",
                    len(self.train_examples_history),
                    " => remove the oldest train_examples",
                )
                self.train_examples_history.pop(0)
            # backup history to a file
            # NB! the examples were collected using the model from the previous iteration, so (i-1)
            self.save_train_examples(i - 1)

            # shuffle examples before training
            train_examples = []
            for e in self.train_examples_history:
                train_examples.extend(e)
            shuffle(train_examples)

            # training new network, keeping a copy of the old one
            self.nnet.save_checkpoint(
                folder=self.args.checkpoint, filename="temp.pth.tar"
            )
            self.pnet.load_checkpoint(
                folder=self.args.checkpoint, filename="temp.pth.tar"
            )
            pmcts = MCTS(self.game, self.pnet, self.args)

            self.nnet.train(train_examples)
            nmcts = MCTS(self.game, self.nnet, self.args)

            print("PITTING AGAINST PREVIOUS VERSION")
            arena = Arena(
                lambda x: np.argmax(pmcts.get_action_prob(x, temp=0)),
                lambda x: np.argmax(nmcts.get_action_prob(x, temp=0)),
                self.game,
            )
            pwins, nwins, draws = arena.play_games(self.args.arenaCompare)

            print(
                "NEW/PREV WINS : %d / %d ; DRAWS : %d" % (nwins, pwins, draws)
            )
            if (
                pwins + nwins == 0
                or float(nwins) / (pwins + nwins) < self.args.updateThreshold
            ):
                print("REJECTING NEW MODEL")
                self.nnet.load_checkpoint(
                    folder=self.args.checkpoint, filename="temp.pth.tar"
                )
            else:
                print("ACCEPTING NEW MODEL")
                self.nnet.save_checkpoint(
                    folder=self.args.checkpoint,
                    filename=self.get_checkpoint_file(i),
                )
                self.nnet.save_checkpoint(
                    folder=self.args.checkpoint, filename="best.pth.tar"
                )

    def get_checkpoint_file(self, iteration):
        return "checkpoint_" + str(iteration) + ".pth.tar"

    def save_train_examples(self, iteration):
        folder = self.args.checkpoint
        if not os.path.exists(folder):
            os.makedirs(folder)
        filename = os.path.join(
            folder, self.get_checkpoint_file(iteration) + ".examples"
        )
        with open(filename, "wb+") as f:
            Pickler(f).dump(self.train_examples_history)

    def load_train_examples(self):
        model_file = os.path.join(
            self.args.load_folder_file[0], self.args.load_folder_file[1]
        )
        examples_file = model_file + ".examples"
        if not os.path.isfile(examples_file):
            print(examples_file)
            r = input("File with training examples not found. Continue? [y|n]")
            if r != "y":
                sys.exit()
        else:
            print("File with training examples found. Read it.")
            with open(examples_file, "rb") as f:
                self.train_examples_history = Unpickler(f).load()
            # examples based on the model were already collected (loaded)
            self.skip_first_self_play = True
