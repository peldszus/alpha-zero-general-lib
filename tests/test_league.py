from alpha_zero_general import League
from alpha_zero_general import RandomPlayer

from example.othello.game import OthelloGame


def test_league_basic_usage():
    game = OthelloGame(6)
    rounds = 2
    league = League(game, initial_rating=1500, rounds=rounds, games=4)
    league.add_player("random_1", RandomPlayer(game), initial_rating=2000)
    league.add_player("random_2", RandomPlayer(game), initial_rating=1750)
    league.add_player("random_3", RandomPlayer(game))
    league.add_player("random_4", RandomPlayer(game), initial_rating=1000)
    assert league.has_started is False
    initial_ratings = league.ratings()
    assert initial_ratings == [
        (2000, 1, "random_1"),
        (1750, 2, "random_2"),
        (1500, 3, "random_3"),
        (1000, 4, "random_4"),
    ]
    league.start()
    assert league.has_started is True
    assert len(league.history) == sum(range(4)) * rounds
    final_ratings = league.ratings()
    assert len(final_ratings) == 4
    assert final_ratings != initial_ratings


def test_league_incrementally_add_players_to_running():
    game = OthelloGame(6)
    rounds = 2
    league = League(game, initial_rating=1500, rounds=rounds, games=4)
    assert league.has_started is False

    league.start()
    assert league.has_started is True
    league.add_player("random_1", RandomPlayer(game))
    assert league.ratings() == [(1500, 1, "random_1")]
    assert len(league.history) == 0

    for i in range(2, 6):
        league.add_player(f"random_{i}", RandomPlayer(game))
        assert len(league.ratings()) == i
        assert len(league.history) == sum(range(i)) * rounds


def test_league_lazy_loading_players():
    game = OthelloGame(6)
    league = League(game, rounds=1, games=2, cache_size=2)
    for i in range(10):
        league.add_player(f"random_{i}", lambda: RandomPlayer(game))
    league.start()
    final_ratings = league.ratings()
    assert len(final_ratings) == 10
