from tictactoe import TicTacToe
from mcts import MCTS
from model import ResNet
import numpy as np
from alphazero import AlphaZero
import torch

tictactoe = TicTacToe()
model = ResNet(tictactoe, 4, 64)
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)


args = {
    'C': 2,
    'num_searches': 60,
    'num_iterations': 3,
    'self_play_iterations': 500,
    'num_epochs': 4,
    'batch_size': 64
}

alphazero = AlphaZero(model, optimizer, tictactoe, args)
alphazero.learn()

# while True:
#     print(state)
#
#     if player == 1:
#         valid_moves = tictactoe.get_valid_moves(state)
#         print('valid_moves', [i for i in range(tictactoe.action_size) if valid_moves[i] == 1])
#         action = int(input(f'{player}:'))
#         if valid_moves[action] == 0:
#             print('action not valid')
#             continue
#     else:
#         neutral_state = tictactoe.change_perspective(state, player)
#         mcts_probs = mcts.search(neutral_state)
#         action = np.argmax(mcts_probs)
#
#
#     state = tictactoe.get_next_state(state, action, player)
#     value, is_terminal = tictactoe.get_value_and_terminated(state, action)
#     if is_terminal:
#         print(state)
#         if value == 1:
#             print(player, 'win')
#         else:
#             print('draw')
#         break
#
#     player = tictactoe.get_opponent(player)
