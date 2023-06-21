import torch
import numpy as np

class DataPoint():
    def __init__(self, board: np.ndarray, mask: list, prob: list, val: float, device:torch.device = None) -> None:
        self.board = torch.tensor(np.array(board), dtype=torch.float32, device=device).unsqueeze(0)
        self.mask = torch.tensor(mask, device=device).unsqueeze(0)
        self.policy = torch.tensor(prob, device=device).unsqueeze(0)
        self.value = torch.tensor(val, device=device).unsqueeze(0).unsqueeze(1)

class DataPoints():
    def __init__(self, max_points: int = 5000, device: torch.device = None) -> None:
        self.data_points = []
        self.num_points = 0
        self.max_points = max_points
        self.device = device

    def add(self, boards, masks, probs, values) -> None:
        num_points = len(boards)
        self.num_points += num_points
        for i in range(num_points):
            dp = DataPoint(boards[i], masks[i], probs[i], values[i], self.device)
            self.data_points.append(dp)

        if self.num_points > self.max_points:
            over = self.num_points - self.max_points
            self.data_points = self.data_points[over:]
            self.num_points += -over

    def get_batch(self, batch_size: int) -> tuple:

        random_points = np.random.choice(self.data_points, batch_size, replace=False)

        tensor_boards = torch.cat([point.board for point in random_points], dim=0).to(self.device)
        tensor_masks = torch.cat([point.mask for point in random_points], dim=0).to(self.device)
        tensor_policies = torch.cat([point.policy for point in random_points], dim=0).to(self.device)
        tensor_values = torch.cat([point.value for point in random_points], dim=0).to(self.device)
        return tensor_boards, tensor_masks, tensor_policies, tensor_values
