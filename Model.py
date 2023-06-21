import torch
from torch import nn, Tensor
from torch.nn import Module

from Game import Connect4

class ResidualBlock(Module):
    def __init__(self, dim: int) -> None:
        super().__init__()

        self.conv1 = nn.Conv2d(in_channels=dim, out_channels=dim, kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(num_features=dim)
        self.conv2 = nn.Conv2d(in_channels=dim, out_channels=dim, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(num_features=dim)

        self.activation = nn.ReLU()

    def forward(self, x: Tensor) -> Tensor:
        out = self.activation(self.bn1(self.conv1(x)))
        out = self.activation(x + self.bn2(self.conv2(out)))
        return out
    
class ConvolutionBlock(Module):
    def __init__(self, in_dim: int, out_dim: int = None, filter_size: int = 3, padding_size: int = 1) -> None:
        super().__init__()

        if out_dim is None:
            out_dim = in_dim

        self.conv = nn.Conv2d(in_dim, out_dim, kernel_size=filter_size, stride=1, padding=padding_size)
        self.bn = nn.BatchNorm2d(out_dim)
        self.activation = nn.ReLU()

    def forward(self, x: Tensor) -> Tensor:
        out = self.activation(self.bn(self.conv(x)))
        return out

class C4Model(Module):
    def __init__(self, dim: int = 128, num_layers: int = 3, policy_dim: int = 2, value_dim: int = 1, value_linear_dim: int = 256) -> None:
        super().__init__()

        self.input_dim = 2
        self.height = 6
        self.width = 7

        self.dim = dim
        self.num_layers = num_layers
        self.policy_dim = policy_dim
        self.value_dim = value_dim
        self.value_linear_dim = value_linear_dim

        self.activation = nn.ReLU()

        self.input_conv = ConvolutionBlock(self.input_dim, self.dim)        

        residual_tower_layers = []
        for i in range(self.num_layers):
            residual_tower_layers.append(ResidualBlock(self.dim))
        self.residual_tower = nn.Sequential(*residual_tower_layers)

        self.policy_head_conv = ConvolutionBlock(self.dim, self.policy_dim, filter_size=1, padding_size=0)
        self.policy_head_linear = nn.Linear(self.policy_dim*self.height*self.width, self.width)
        self.policy_head_softmax = nn.Softmax(dim=1)

        self.value_head_conv = ConvolutionBlock(self.dim, self.value_dim, filter_size=1, padding_size=0)
        self.value_head_linear = nn.Sequential(
            nn.Linear(self.value_dim*self.height*self.width, self.value_linear_dim),
            self.activation,
            nn.Linear(self.value_linear_dim, 1),
            nn.Tanh()
        )

    def forward(self, x: Tensor, p_mask: Tensor = None) -> tuple:
        # p_mask: 0 for no mask, 1 for -inf mask

        out = self.input_conv(x)
        out = self.residual_tower(out)
        policy = self.policy_head_conv(out)
        value = self.value_head_conv(out)

        batch_size = x.size(0)
        policy = self.policy_head_linear(policy.view(batch_size, -1))
        value = self.value_head_linear(value.view(batch_size, -1))

        if not (p_mask is None):
            mask = self.generate_mask(p_mask)
            policy += mask
        policy = self.policy_head_softmax(policy)

        return policy, value
    
    def generate_mask(self, p_mask: Tensor = None) -> Tensor:
        return torch.where(p_mask, float('-inf'), 0)


def loss_function(pred_policy: Tensor, pred_value: Tensor, target_policy: Tensor, target_value: Tensor) -> Tensor:
    mse = (target_value-pred_value).pow(2).sum()
    ce = -(target_policy * (pred_policy+1e-40).log()).sum()
    return mse + ce

class Evaluator():
    def __init__(self, model: C4Model, device: torch.device = None) -> None:
        self.model = model
        self.device = device

    def __call__(self, x: Connect4) -> tuple:
        if x.get_is_terminal():
            return [0]*7, x.value
        board = x.board.copy()
        mask = x.get_impossible_actions()

        with torch.no_grad():
            tensor_board = torch.from_numpy(board).to(self.device).float().unsqueeze(0)
            tensor_mask = torch.tensor(mask, dtype=torch.bool, device=self.device).unsqueeze(0)
            policy, value = self.model(tensor_board, tensor_mask)
        return policy[0].tolist(), value.item()

