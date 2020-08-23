"""A league where players can compete against each other and be ranked with the Elo rating system."""

from collections import namedtuple
from itertools import combinations

from alpha_zero_general import Arena
from alpha_zero_general import Player
from cachetools import LRUCache
from elote import EloCompetitor

MatchResult = namedtuple(
    "MatchResult",
    [
        "player_id_a",
        "player_id_b",
        "rating_a",
        "rating_b",
        "wins",
        "losts",
        "ties",
        "rating_change_a",
        "rating_change_b",
    ],
)


class League:
    def __init__(
        self, game, initial_rating=1500, rounds=1, games=4, cache_size=10
    ):
        """
        A league where players can compete against each other and be ranked
        with the Elo rating system.
        """
        self.game = game
        self.initial_rating = initial_rating
        self.rounds = rounds
        self.games = games
        self.has_started = False
        self.competitors = {}
        self.history = []
        self._cache = LRUCache(cache_size)

    def add_player(self, name, player, initial_rating=None):
        """
        Adds a player to the league.

        The `player` can either be an instance of `alpha_zero_general.Player`.
        Alternatively the `player` can a function that returns a player object,
        in order to be able to lazily load the player only when needed.

        If the league already started, adding a new player will trigger matches
        of the new player against all existing players for `self.rounds`, thus
        allowing incremental league competition.
        """
        player_id = self._register_player(
            name, player, initial_rating=initial_rating
        )

        if self.has_started:
            for _round_nr in range(1, self.rounds + 1):
                for competitor in sorted(self.competitors):
                    if competitor != player_id:
                        self._match(player_id, competitor)

        return player_id

    def start(self):
        """Run the league with all registered players."""
        self.has_started = True
        for _round_nr in range(1, self.rounds + 1):
            for a, b in sorted(combinations(self.competitors, 2)):
                self._match(a, b)

    def ratings(self):
        """Returns the elo ratings of all competitors in the league."""
        ratings = [
            (elo.rating, player_id, name)
            for player_id, (name, _, elo) in self.competitors.items()
        ]
        return sorted(ratings, reverse=True)

    def _register_player(self, name, player, initial_rating=None):
        """Add a player """
        player_id = len(self.competitors) + 1
        elo = EloCompetitor(
            initial_rating=initial_rating or self.initial_rating
        )
        self.competitors[player_id] = name, player, elo
        return player_id

    def _get_competitor(self, player_id):
        """
        Returns the name, the player instance and the elo object for the
        given player_id. If the player was to be lazyloaded by a function,
        and is not yet or not anymore in the cache, the instance is created
        and managed in the cache.
        """
        name, player_or_function, elo = self.competitors[player_id]
        if player_id in self._cache:
            player = self._cache[player_id]
        elif isinstance(player_or_function, Player):
            player = player_or_function
        else:
            player = player_or_function()
            self._cache[player_id] = player
        return name, player, elo

    def _match(self, a, b):
        """Play one match of player_id `a` vs `b`."""
        name_a, player_a, elo_a = self._get_competitor(a)
        name_b, player_b, elo_b = self._get_competitor(b)
        rating_a = elo_a.rating
        rating_b = elo_b.rating
        arena = Arena(player_a, player_b, self.game)
        wins, losts, ties = arena.play_games(
            self.games, verbose=False, quiet=True
        )
        print(name_a, name_b, wins, losts, ties)
        if wins > losts:
            elo_a.beat(elo_b)
        elif losts > wins:
            elo_b.beat(elo_a)
        else:
            elo_a.tied(elo_b)
        rating_change_a = elo_a.rating - rating_a
        rating_change_b = elo_b.rating - rating_b
        self.history.append(
            MatchResult(
                a,
                b,
                rating_a,
                rating_b,
                wins,
                losts,
                ties,
                rating_change_a,
                rating_change_b,
            )
        )
