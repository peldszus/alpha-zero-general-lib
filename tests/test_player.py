from unittest.mock import patch

import pytest
from alpha_zero_general import AlphaZeroPlayer
from alpha_zero_general import BareModelPlayer
from alpha_zero_general import GreedyPlayer
from alpha_zero_general import HumanPlayer
from alpha_zero_general import RandomPlayer

from example.othello.game import OthelloGame
from example.othello.keras import OthelloNNet


def test_alpha_zero_player():
    game = OthelloGame(6)
    player = AlphaZeroPlayer(game, OthelloNNet, num_mcts_sims=4)
    board = game.get_init_board()
    action = player.play(board)
    assert action


def test_alpha_zero_player_from_checkpoint():
    game = OthelloGame(6)

    some_net = OthelloNNet(game)
    folder, filename = "/tmp/", "checkpoint_alpha_zero_player"
    some_net.save_checkpoint(folder, filename)
    del some_net

    player = AlphaZeroPlayer(
        game, OthelloNNet, folder=folder, filename=filename, num_mcts_sims=4
    )
    board = game.get_init_board()
    action = player.play(board)
    assert action


def test_alpha_zero_player_from_model():
    game = OthelloGame(6)
    some_net = OthelloNNet(game)
    player = AlphaZeroPlayer(game, some_net)
    board = game.get_init_board()
    action = player.play(board)
    assert action


def test_alpha_zero_player_reset():
    game = OthelloGame(6)
    player = AlphaZeroPlayer(game, OthelloNNet, num_mcts_sims=4)
    assert not player.mcts.Qsa
    board = game.get_init_board()
    player.play(board)
    assert player.mcts.Qsa
    player.reset()
    assert not player.mcts.Qsa


def test_bare_model_player():
    game = OthelloGame(6)
    player = BareModelPlayer(game, OthelloNNet)
    board = game.get_init_board()
    action = player.play(board)
    assert action


def test_bare_model_player_from_checkpoint():
    game = OthelloGame(6)

    some_net = OthelloNNet(game)
    folder, filename = "/tmp/", "checkpoint_bare_model_player"
    some_net.save_checkpoint(folder, filename)
    del some_net

    player = BareModelPlayer(
        game, OthelloNNet, folder=folder, filename=filename
    )
    board = game.get_init_board()
    action = player.play(board)
    assert action


def test_bare_model_player_from_model():
    game = OthelloGame(6)
    some_net = OthelloNNet(game)
    player = BareModelPlayer(game, some_net)
    board = game.get_init_board()
    action = player.play(board)
    assert action


def test_bare_model_player_invalid_nnet_parameter():
    neither_nnet_nor_nnet_class = "something_else"
    game = OthelloGame(6)
    with pytest.raises(TypeError) as excinfo:
        BareModelPlayer(game, neither_nnet_nor_nnet_class)
    assert "NeuralNet subclass or instance" in str(excinfo.value)


def test_greedy_player():
    game = OthelloGame(6)
    player = GreedyPlayer(game)
    board = game.get_init_board()
    action = player.play(board)
    assert action


def test_random_player():
    game = OthelloGame(6)
    player = RandomPlayer(game)
    board = game.get_init_board()
    action = player.play(board)
    assert action


def test_human_player_valid_action():
    action_valid = "1,2"
    game = OthelloGame(6)
    player = HumanPlayer(game)
    board = game.get_init_board()
    with patch("builtins.input", side_effect=[action_valid]):
        action = player.play(board)
    assert action == 8


def test_human_player_unknown_action(capsys):
    action_valid = "1,2"
    action_unknown = "x,x"
    game = OthelloGame(6)
    player = HumanPlayer(game)
    board = game.get_init_board()
    with patch("builtins.input", side_effect=[action_unknown, action_valid]):
        player.play(board)
    out, _err = capsys.readouterr()
    assert out == "Unknown action.\n"


def test_human_player_invalid_action(capsys):
    action_valid = "1,2"
    action_invalid = "0,0"
    game = OthelloGame(6)
    player = HumanPlayer(game)
    board = game.get_init_board()
    with patch("builtins.input", side_effect=[action_invalid, action_valid]):
        player.play(board)
    out, _err = capsys.readouterr()
    assert out == "Invalid action.\n"
