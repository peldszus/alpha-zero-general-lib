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
    print(arena.play_games(10, verbose=True))
