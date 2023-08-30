import numpy as np


class TicTacToe:
    def __init__(self):
        self.row_count = 3
        self.col_count = 3
        self.action_size = self.col_count * self.row_count

    def get_initial_state(self):
        return np.zeros((self.row_count, self.col_count))

    def get_next_state(self, state, action, player):
        row = action // self.col_count
        col = action % self.col_count
        state[row, col] = int(player)
        return state

    @staticmethod
    def get_valid_moves(state):
        return (state.reshape(-1) == 0).astype(np.uint8)

    def check_win(self, state, action):
        if action is None:
            return False

        row = action // self.col_count
        col = action % self.col_count
        player = state[row, col]

        return (
            np.sum(state[row, :]) == player * self.col_count
            or np.sum(state[:, col]) == player * self.row_count
            or np.sum(np.diag(state)) == player * self.row_count
            or np.sum(np.diag(np.flip(state, axis=0))) == player * self.row_count
        )

    def get_value_and_terminated(self, state, action):
        if self.check_win(state, action):
            return 1, True
        if np.sum(self.get_valid_moves(state)) == 0:
            return 0, True

        return 0, False

    @staticmethod
    def get_opponent(player):
        return -player

    @staticmethod
    def get_opponent_value(value):
        return -value

    @staticmethod
    def change_perspective(state, player):
        return state * player
