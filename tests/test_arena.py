from alpha_zero_general import Arena
from alpha_zero_general import RandomPlayer

from example.othello.game import OthelloGame


def test_arena():
    game = OthelloGame(6)
    player1 = RandomPlayer(game)
    player2 = RandomPlayer(game)
    arena = Arena(
        player1.play, player2.play, game, display=OthelloGame.display
    )

    number_of_games = 10
    one_won, two_won, draws = arena.play_games(number_of_games, verbose=True)
    assert one_won + two_won + draws == number_of_games
