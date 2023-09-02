from tictactoe import TicTacToe
from connect_four import ConnectFour
from model import ResNet
from alphazero import AlphaZero
import torch
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# can be changed to tictactoe
game = TicTacToe()

# use a simpler network for tictactoe
model = ResNet(game, 4, 64, device=device)

optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=0.0001)

args = {
    'C': 2,
    'num_searches': 60,
    'num_iterations': 6,
    'self_play_iterations': 500,
    'num_epochs': 8,
    'batch_size': 64,
    'temperature': 1.25,
    'dirichlet_epsilon': 0.25,
    'dirichlet_alpha': 0.3,
}

alphazero = AlphaZero(model, optimizer, game, args)
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
