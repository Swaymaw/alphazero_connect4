import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt


class ResBlock(nn.Module):
    def __init__(self, num_hidden):
        super().__init__()
        self.conv1 = nn.Conv2d(num_hidden, num_hidden, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(num_hidden)
        self.conv2 = nn.Conv2d(num_hidden, num_hidden, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(num_hidden)

    def forward(self, x):
        residual = x
        x = self.conv1(x)
        x = self.bn1(x)
        x = F.relu(x)
        x = self.conv2(x)
        x = self.bn2(x)
        return F.relu(x + residual)


class ResNet(nn.Module):
    def __init__(self, game, num_resBlocks, num_hidden, device):
        super().__init__()
        self.device = device

        self.startBlock = nn.Sequential(
            nn.Conv2d(3, num_hidden, kernel_size=3, padding=1),
            nn.BatchNorm2d(num_hidden),
            nn.ReLU(),
        )

        self.backbone = nn.ModuleList(
            [ResBlock(num_hidden) for _ in range(num_resBlocks)]
        )

        self.policyhead = nn.Sequential(
            nn.Conv2d(num_hidden, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(32 * game.row_count * game.col_count, game.action_size),
        )

        self.valuehead = nn.Sequential(
            nn.Conv2d(num_hidden, 3, kernel_size=3, padding=1),
            nn.BatchNorm2d(3),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(3 * game.row_count * game.col_count, 1),
            nn.Tanh()
        )

        self.to(device)

    def forward(self, x):
        x = self.startBlock(x)
        for resBlock in self.backbone:
            x = resBlock(x)

        policy = self.policyhead(x)
        value = self.valuehead(x)

        return policy, value


def test():
    from tictactoe import TicTacToe
    game = TicTacToe()
    state = game.get_initial_state()
    state = game.get_next_state(state, 2, 1)
    state = game.get_next_state(state, 4, -1)
    state = game.get_next_state(state, 8, 1)
    state = game.get_next_state(state, 1, -1)

    print(state)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    encoded_state = game.get_encoded_state(state)
    encoded_state = torch.tensor(encoded_state, device=device).view(-1, encoded_state.shape[0],
                                                                    encoded_state.shape[1], encoded_state.shape[2])

    res_net = ResNet(game, 4, 64, device=device)
    res_net.load_state_dict(torch.load(f'model_2_{game}.pt'))
    res_net.eval()

    policy, value = res_net(encoded_state)
    policy = torch.softmax(policy, axis=1).squeeze().detach().to('cpu').numpy()
    # print(f'Policy: {policy}\nValue: {value.tolist()[0]}')
    plt.bar(range(game.action_size), policy)

    plt.show()


if __name__ == '__main__':
    test()
