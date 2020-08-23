"""An Arena class where any 2 agents can be pit against each other."""

from tqdm import tqdm


class Arena:
    """
    An Arena class where any 2 agents can be pit against each other.
    """

    def __init__(self, player1, player2, game, display=None):
        """
        Input:
            player 1,2: two instances of a Player class.
            game: Game object
            display: a function that takes board as input and prints it (e.g.
                     display in othello/OthelloGame). Is necessary for verbose
                     mode.
        """
        self.player1 = player1
        self.player2 = player2
        self.game = game
        self.display = display

    def play_game(self, verbose=False):
        """
        Executes one episode of a game.

        Returns:
            either
                winner: player who won the game (1 if player1, -1 if player2)
            or
                draw result returned from the game that is neither 1, -1, nor 0.
        """
        self.reset_players()
        players = [self.player2, None, self.player1]
        current_player = 1
        board = self.game.get_init_board()
        it = 0
        while self.game.get_game_ended(board, current_player) == 0:
            it += 1
            if verbose:
                assert self.display
                print("Turn ", str(it), "Player ", str(current_player))
                self.display(board)

            action = players[current_player + 1].play(
                self.game.get_canonical_form(board, current_player)
            )

            valids = self.game.get_valid_moves(
                self.game.get_canonical_form(board, current_player), 1
            )

            if valids[action] == 0:
                print(action)
                assert valids[action] > 0
            board, current_player = self.game.get_next_state(
                board, current_player, action
            )
        if verbose:
            assert self.display
            print(
                "Game over: Turn ",
                str(it),
                "Result ",
                str(self.game.get_game_ended(board, 1)),
            )
            self.display(board)
        return current_player * self.game.get_game_ended(board, current_player)

    def play_games(self, num, verbose=False, quiet=False):
        """
        Plays num games in which player1 starts num/2 games and player2 starts
        num/2 games.

        Returns:
            oneWon: games won by player1
            twoWon: games won by player2
            draws:  games won by nobody
        """

        num = int(num / 2)
        one_won = 0
        two_won = 0
        draws = 0
        for _ in tqdm(
            range(num), desc="Arena.play_games (Player 1)", disable=quiet
        ):
            game_result = self.play_game(verbose=verbose)
            if game_result == 1:
                one_won += 1
            elif game_result == -1:
                two_won += 1
            else:
                draws += 1

        self.player1, self.player2 = self.player2, self.player1

        for _ in tqdm(
            range(num), desc="Arena.play_games (Player 2)", disable=quiet
        ):
            game_result = self.play_game(verbose=verbose)
            if game_result == -1:
                one_won += 1
            elif game_result == 1:
                two_won += 1
            else:
                draws += 1

        return one_won, two_won, draws

    def reset_players(self):
        """Reset the players."""
        self.player1.reset()
        self.player1.reset()
