# Alpha Zero General - as a library

[![Build Status](https://travis-ci.org/peldszus/alpha-zero-general-lib.svg?branch=master)](https://travis-ci.org/peldszus/alpha-zero-general-lib)
[![codecov](https://codecov.io/gh/peldszus/alpha-zero-general-lib/branch/master/graph/badge.svg)](https://codecov.io/gh/peldszus/alpha-zero-general-lib)
[![GitHub](https://img.shields.io/github/license/peldszus/alpha-zero-general-lib)](LICENSE)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/ambv/black)


An implementation of the [AlphaZero](https://doi.org/10.1126/science.aar6404) algorithm for adversarial games to be used with the machine learning framework of your choice.

This is a fork of https://github.com/suragnair/alpha-zero-general turned into a library.

The idea is to have a small and clean library version of the code, with minimal requirements, without many implementations of specific games or large model files. The only 'heavy' requirement is the [ray](https://github.com/ray-project/ray/)-library, which is used to make the algorithm fully async and parallelized (potentially even across multiple machines).


## Information

A simplified, highly flexible, commented and (hopefully) easy to understand implementation of the self-play based reinforcement learning algorithm Alpha Zero. It is designed to be easy to adopt for any two-player turn-based adversarial game with perfect information. Use it with the machine learning framework you like. A sample implementation has been provided for the game of Othello in TensorFlow/Keras, see `example/othello/`.


## Usage

To use this library for the game of your choice, subclass from `Game` and `NeuralNet` and implement their functions. Define your config and use the `Coach` to start the learning algorithm. Test your model by playing against it in the `Arena` against one of the `Player` classes. Evaluate multiple players in a `League` and let it calculate the ELO score for you.


## Motivation

Why did I do this? Well, mostly for myself. :) I wanted to play with the algorithm, I was in a mood of working with existing code, and I wanted to learn about ray.


## ToDo list

**Library:**
* [x] Poetry based build that works as library
* [x] Add some tests
* [x] Use travis-ci.org tests
* [x] Coverage report on codecov.org
* [x] Make sure it works
* [ ] More documentation
* [ ] Provide one entrypoint for training and pitting
* [ ] A first pypi release
* [ ] Work on remaining test coverage gaps
* [ ] Add fixtures in conftest, for game net and tempdir

**Refactor:**
* [x] Use `tqdm` for progress bars
* [x] Pythonic renaming of modules
* [x] Pythonic renaming of method names
* [x] Pythonic renaming of variables names
* [x] Black formatting, flake8
* [x] Fix inline comments
* [x] Fix all flake8 issues
* [x] Proper abstract classes for Game and NeuralNet
* [ ] Make MCTS and model parameters explicit
* [ ] .. or replace DotDict with overall config class
* [ ] Use logging
* [ ] Remove obsolete parameters
* [ ] Add game-based parameters

**General player classes:**
* [x] AlphaZeroPlayer out of pit code
* [x] BareModelPlayer
* [x] HumanPlayer: named action mapping

**Asynchronous & parallel processing:**
* [x] Ray step 1: Use Ray to parallelize self-play
* [x] Ray step 2: Share weights across ray actors
* [x] Ray step 3: Make self-play and training fully async
* [x] Add parameter to control selfplay vs training
* [ ] Ray step 4: Parallelize the arena play during league execution
* [ ] Successfully try multi-machine parallelization

**Improvements:**
* [x] Store all models, if accepted or not
* [x] Store training examples per game to reduce data duplication
* [x] Be able to continue training
* [ ] Add dirichlet noise for better exploration

**New features:**
* [x] League evaluations with ELO scores


## Develop

Requirements:
* Operating system: Linux/Mac (Windows is only experimental in the ray-library)
* Python >= 3.7


To locally build and install, simply run
```
make
```

To execute the tests, run
```
make test
```

This will additionally install tensorflow, because the keras example implementation of Othello is used during the tests.


## Evaluate

How can I know a change in the code/a change of parameters is actually for the good? How can I evaluate that this brings better results?

* First rule: Only change one parameter at a time when comparing two runs.
* Random choice is also a parameter: Be sure to set the same random seeds across runs: for python, for numpy, and for your framework, like tensorflow/pytorch.
* Repeat your experiment with different random seeds.
* Initial model parameters are parameters: Start from the same initialized (untrained) model across runs.
* Be aware that changing e.g. exploration parameters might have a different impact in different phases of the training. Ideally, you have an 'early' game model (where the model has only seen a none to few games), a 'mid' game (where it has seen several thousand games) and a 'late' game model (which as seen a lot of games). Observe the effect of your change of code/parameters in all three stages.
* Don't compare the model training losses. Since the training data is continuously changing, you wouldn't have a common ground for comparing those.
* Compare the game play performance:
  * Let two competitor agents play against each other in the Arena (remember this requires that changes to the code need to be fully parameterized).
  * Let two competitor agents play against baselines (like `RandomPlayer`, `GreedyPlayer`, `BareModelPlayer`).
  * Observe the win rate or the ELO in a tournament.


## Contributors and Credits
* The original version was written by [Surag Nair](https://github.com/suragnair) and credits go to all contributors of https://github.com/suragnair/alpha-zero-general.
* The use of ray is inspired by https://github.com/werner-duvaud/muzero-general.
