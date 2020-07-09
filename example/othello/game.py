import numpy as np
from alpha_zero_general import Game

from .logic import Board


class OthelloGame(Game):
    square_content = {-1: "X", +0: "-", +1: "O"}

    @staticmethod
    def get_square_piece(piece):
        return OthelloGame.square_content[piece]

    def __init__(self, n):
        self.n = n
        self.action_names = {
            f"{x},{y}": x * n + y for x in range(n) for y in range(n)
        }

    def get_init_board(self):
        # return initial board (numpy board)
        b = Board(self.n)
        return np.array(b.pieces)

    def get_board_size(self):
        # (a,b) tuple
        return (self.n, self.n)

    def get_action_size(self):
        # return number of actions
        return self.n * self.n + 1

    def get_action_names(self):
        return self.action_names

    def get_action_prompt(self):
        return "Your move: 'row,col' > "

    def get_next_state(self, board, player, action):
        # if player takes action on board, return next (board,player)
        # action must be a valid move
        if action == self.n * self.n:
            return (board, -player)
        b = Board(self.n)
        b.pieces = np.copy(board)
        move = (int(action / self.n), action % self.n)
        b.execute_move(move, player)
        return (b.pieces, -player)

    def get_valid_moves(self, board, player):
        # return a fixed size binary vector
        valids = [0] * self.get_action_size()
        b = Board(self.n)
        b.pieces = np.copy(board)
        legal_moves = b.get_legal_moves(player)
        if len(legal_moves) == 0:
            valids[-1] = 1
            return np.array(valids)
        for x, y in legal_moves:
            valids[self.n * x + y] = 1
        return np.array(valids)

    def get_game_ended(self, board, player):
        # return 0 if not ended, 1 if player 1 won, -1 if player 1 lost
        # player = 1
        b = Board(self.n)
        b.pieces = np.copy(board)
        if b.has_legal_moves(player):
            return 0
        if b.has_legal_moves(-player):
            return 0
        if b.count_diff(player) > 0:
            return 1
        return -1

    def get_canonical_form(self, board, player):
        # return state if player==1, else return -state if player==-1
        return player * board

    def get_symmetries(self, board, pi):
        # mirror, rotational
        assert len(pi) == self.n ** 2 + 1  # 1 for pass
        pi_board = np.reshape(pi[:-1], (self.n, self.n))
        result = []

        for i in range(1, 5):
            for j in [True, False]:
                newB = np.rot90(board, i)
                newPi = np.rot90(pi_board, i)
                if j:
                    newB = np.fliplr(newB)
                    newPi = np.fliplr(newPi)
                result += [(newB, list(newPi.ravel()) + [pi[-1]])]
        return result

    def string_representation(self, board):
        return board.tobytes()

    def string_representation_readable(self, board):
        board_s = "".join(
            self.square_content[square] for row in board for square in row
        )
        return board_s

    def get_score(self, board, player):
        b = Board(self.n)
        b.pieces = np.copy(board)
        return b.count_diff(player)

    @staticmethod
    def display(board):
        n = board.shape[0]
        print("   ", end="")
        for y in range(n):
            print(y, end=" ")
        print("")
        print("-----------------------")
        for y in range(n):
            print(y, "|", end="")  # print the row #
            for x in range(n):
                piece = board[y][x]  # get the piece to print
                print(OthelloGame.square_content[piece], end=" ")
            print("|")

        print("-----------------------")
