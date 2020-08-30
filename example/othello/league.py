from functools import partial

from alpha_zero_general import AlphaZeroPlayer
from alpha_zero_general import BareModelPlayer  # noqa
from alpha_zero_general import GreedyPlayer
from alpha_zero_general import League
from alpha_zero_general import RandomPlayer

from .game import OthelloGame
from .pytorch import OthelloNNet

game = OthelloGame(6)

folder, filename = "./runs/1", "model_00100"
random = RandomPlayer(game)
greedy = GreedyPlayer(game)
# bare = BareModelPlayer(game, OthelloNNet, folder, filename)
# alpha = AlphaZeroPlayer(game, OthelloNNet, folder, filename)

league = League(game, cache_size=3)
league.start()
league.add_player("random", random)
league.add_player("greedy", greedy)
# league.add_player("bare", bare)
# league.add_player("alpha", alpha)


def alpha_zero_at_checkpoint(i):
    return AlphaZeroPlayer(game, OthelloNNet, "./runs/1", f"model_{i:05d}")


for i in [10, 20, 50, 100, 150, 200]:
    league.add_player(f"alpha-{i}", partial(alpha_zero_at_checkpoint, i))

print(league.ratings())
print(league.history)
