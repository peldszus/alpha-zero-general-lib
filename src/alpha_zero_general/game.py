"""An abstract base class for games to be played by alpha zero."""

from abc import ABC
from abc import abstractmethod


class Game(ABC):
    """
    This class specifies the base Game class. To define your own game, subclass
    this class and implement the functions below. This works when the game is
    two-player, adversarial and turn-based.

    Use 1 for player1 and -1 for player2.
    """

    @abstractmethod
    def get_init_board(self):
        """
        Returns:
            startBoard: a representation of the board (ideally this is the form
                        that will be the input to your neural network)
        """

    @abstractmethod
    def get_board_size(self):
        """
        Returns:
            (x,y): a tuple of board dimensions
        """

    @abstractmethod
    def get_action_size(self):
        """
        Returns:
            actionSize: number of all possible actions
        """

    @abstractmethod
    def get_action_names(self):
        """
        Returns:
            action_names: a dictionary mapping action names to actions
        """

    def get_action_prompt(self):
        """
        Returns:
            action_prompt: A message shown to ask a human player for input.
        """
        return "Your move > "

    @abstractmethod
    def get_next_state(self, board, player, action):
        """
        Input:
            board: current board
            player: current player (1 or -1)
            action: action taken by current player

        Returns:
            nextBoard: board after applying action
            nextPlayer: player who plays in the next turn (should be -player)
        """

    @abstractmethod
    def get_valid_moves(self, board, player):
        """
        Input:
            board: current board
            player: current player

        Returns:
            validMoves: a binary vector of length self.get_action_size(), 1 for
                        moves that are valid from the current board and player,
                        0 for invalid moves
        """

    @abstractmethod
    def get_game_ended(self, board, player):
        """
        Input:
            board: current board
            player: current player (1 or -1)

        Returns:
            r: 0 if game has not ended. 1 if player won, -1 if player lost,
               small non-zero value for draw.
        """

    @abstractmethod
    def get_canonical_form(self, board, player):
        """
        Input:
            board: current board
            player: current player (1 or -1)

        Returns:
            canonicalBoard: returns canonical form of board. The canonical form
                            should be independent of player. For e.g. in chess,
                            the canonical form can be chosen to be from the pov
                            of white. When the player is white, we can return
                            board as is. When the player is black, we can invert
                            the colors and return the board.
        """

    @abstractmethod
    def get_symmetries(self, board, pi):
        """
        Input:
            board: current board
            pi: policy vector of size self.get_action_size()

        Returns:
            symmForms: a list of [(board,pi)] where each tuple is a symmetrical
                       form of the board and the corresponding pi vector. This
                       is used when training the neural network from examples.
        """

    @abstractmethod
    def string_representation(self, board):
        """
        Input:
            board: current board

        Returns:
            boardString: a quick conversion of board to a string format.
                         Required by MCTS for hashing.
        """
