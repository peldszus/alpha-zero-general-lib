# Alpha Zero General - as a library

[![Build Status](https://travis-ci.org/peldszus/alpha-zero-general-lib.svg?branch=master)](https://travis-ci.org/peldszus/alpha-zero-general-lib)
[![codecov](https://codecov.io/gh/peldszus/alpha-zero-general-lib/branch/master/graph/badge.svg)](https://codecov.io/gh/peldszus/alpha-zero-general-lib)
[![GitHub](https://img.shields.io/github/license/peldszus/alpha-zero-general-lib)](LICENSE)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/ambv/black)

This is a fork of https://github.com/suragnair/alpha-zero-general turned into a library.

The idea is to have a small and clean library version of the code, with minimal requirements, without many implementations of specific games or large model files.


## Information

A simplified, highly flexible, commented and (hopefully) easy to understand implementation of self-play based reinforcement learning based on the AlphaGo Zero paper (Silver et al). It is designed to be easy to adopt for any two-player turn-based adversarial game and any deep learning framework of your choice. A sample implementation has been provided for the game of Othello in TensorFlow/Keras, see `example/othello/`.

To use this library for a game of your choice, subclass from `alpha_zero_general.Game` and `alpha_zero_general.NeuralNet` and implement their functions.

`alpha_zero_general.Coach` contains the core training loop and ```alpha_zero_general.MCTS``` performs the Monte Carlo Tree Search.

## TODO list

**Library:**
* [x] Poetry based build that works as library
* [x] Add some tests
* [x] Use travis-ci.org tests
* [x] Coverage report on codecov.org
* [x] Make sure it works
* [ ] More documentation
* [ ] Provide one entrypoint for training and pitting
* [ ] A first pypi release

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

**General player classes:**
* [x] AlphaZeroPlayer out of pit code
* [x] BareModelPlayer
* [x] HumanPlayer: named action mapping

**Asynchronous & parallel processing:**
* [x] Ray step 1: Use Ray to parallelize self-play
* [x] Ray step 2: Share weights across ray actors (for multi-machine parallelization)
* [x] Ray step 3: Make self-play and training fully async
* [ ] Ray step 4: Parallelize the arena play during league execution
* [x] Add parameter to control selfplay vs training

**Improvements:**
* [x] Store all models, if accepted or not
* [x] Store training examples per game to reduce data duplication
* [x] Be able to continue training
* [ ] Add dirichlet noise for better exploration

**New features:**
* [x] League evaluations with ELO scores



## Contributors and Credits
* The original version was written by [Surag Nair](https://github.com/suragnair) and credits go to all contributors of https://github.com/suragnair/alpha-zero-general.
* The use of ray is inspired by https://github.com/werner-duvaud/muzero-general.
