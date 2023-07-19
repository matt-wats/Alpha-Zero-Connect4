import numpy as np


class Connect4():
    def __init__(self, height: int = 6, width: int = 7, board: np.ndarray = None, moves: list = None, value: float = None, flipped: bool = False) -> None:
        self.height = height
        self.width = width

        self.board = np.zeros((2, height, width), dtype=bool) if board is None else board
        self.column_moves = [height - 1] * width if moves is None else moves

        self.value = value  # None if game continues, 0.0 for draw, 1 for win
        self.flipped = flipped
        self.is_terminal = False

    def copy(self):
        # Create a copy of Connect4 object
        board_copy = Connect4(height=self.height, width=self.width, board=self.board.copy(),
                              moves=self.column_moves.copy(), value=self.value, flipped=self.flipped)
        return board_copy

    def do_move(self, col: int): # -> Connect4
        assert self.value is None, "Can't Move After Game Ends"

        # Get row position and check if valid
        row = self.column_moves[col]
        assert row > -1, "Not a Valid Move"

        # Place piece
        self.board[0, row, col] = True

        # Update column's row position and flip board
        self.column_moves[col] -= 1
        self.board = np.flip(self.board, axis=0)
        self.flipped = not self.flipped

        # Check game conditions
        self.value = self.update_value(row, col)
        self.is_terminal = (self.value is not None)

        return self

    def update_value(self, row: int, col: int) -> float:
        # Check if game is over
        assert self.value is None, "Can't Check for Win After Game is Over"

        if max(self.column_moves) == -1:
            return 0.0

        row_min = max(0, row - 3)
        row_max = min(self.height - 1, row + 3)
        col_min = max(0, col - 3)
        col_max = min(self.width - 1, col + 3)

        # Vertical connect 4s
        if self.is_connect(self.board[1, row_min:row_max + 1, col]):
            return 1.0

        # Horizontal connect 4s
        if self.is_connect(self.board[1, row, col_min:col_max + 1]):
            return 1.0

        # Diagonal connect 4s
        top_left_length = min(row - row_min, col - col_min)
        bottom_right_length = min(row_max - row, col_max - col)
        top_right_length = min(row - row_min, col_max - col)
        bottom_left_length = min(row_max - row, col - col_min)

        # \
        ar = np.arange(-top_left_length, bottom_right_length + 1)
        if self.is_connect(self.board[1, row + ar, col + ar]):
            return 1.0

        # /
        ar = np.arange(-top_right_length, bottom_left_length + 1)
        if self.is_connect(self.board[1, row + ar, col - ar]):
            return 1.0

        return None

    def is_connect(self, seq):
        # Given sequence, return if 4 in a row
        if len(seq) < 4:
            return False
        for i in range(len(seq) - 3):
            if np.min(seq[i:i + 4]) == 1:
                return True
        return False

    def get_valid_moves(self) -> list:
        # returns columns that pieces can be placed in
        return [col for col, row in enumerate(self.column_moves) if row > -1] if self.value is None else []

    def get_legal_states(self) -> tuple:
        # returns all next possible positions of current board
        valid_moves = self.get_valid_moves()
        legal_states = [self.copy().do_move(col) for col in valid_moves]
        return legal_states, valid_moves

    def get_is_terminal(self) -> bool:
        return self.is_terminal

    def get_impossible_actions(self) -> list:
        return [row == -1 for row in self.column_moves] if not self.get_is_terminal() else [False for row in self.column_moves]
