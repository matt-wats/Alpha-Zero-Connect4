import torch
import numpy as np
from Model import C4Model, Evaluator, loss_function
from Simulate_Game import simulate_game
from Data import DataPoints

from matplotlib import pyplot as plt

from tqdm import tqdm

import os

device = "cpu" #torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Initialize seeds, models, and optimizer
torch.manual_seed(42)
np.random.seed(42)
model = C4Model().to(device)
optimizer = torch.optim.SGD(params=model.parameters(), lr=1e-4, momentum=0.9, weight_decay=1e-6)
evaluator = Evaluator(model, device)
losses = []

# Initialize class that holds new data
data = DataPoints(max_points=5_000, device=device)


# Train loop
num_games = 2_000
batch_size = 32
num_batches = 10

for g in tqdm(range(num_games)):
    with torch.no_grad():
        _, _, boards, masks, probs, values = simulate_game(evaluator, search_iters=100)
        data.add(boards, masks, probs, values)

    for epoch in range(num_batches):
        if data.num_points < batch_size:
            break
        batch_boards, batch_masks, batch_policies, batch_values = data.get_batch(batch_size)

        optimizer.zero_grad()
        pred_policies, pred_values = model(batch_boards, batch_masks)

        loss = loss_function(pred_policies, pred_values, batch_policies, batch_values)
        loss.backward()
        optimizer.step()
        losses.append(loss.item()/batch_size)

# Save model
folder_path = "./Trained_Model/"
if not os.path.exists(folder_path):
    os.makedirs(folder_path)
torch.save(model, folder_path + "model.pt")
torch.save(model.state_dict(), folder_path + "model_state.pt")
np.save(folder_path + "losses.npy", losses)


# Plot losses
plt.plot(losses)
plt.show()

plt.plot([np.mean(losses[i:i+10]) for i in range(0,len(losses),10)])
plt.show()