import os
os.environ.pop("MPLBACKEND", None)

import random
import pickle
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from clearml import Task, Logger
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

task = Task.init(
    project_name="AI_Studio_Basic_Demo",
    task_name="Pipeline step 3 train model"
)
logger = Logger.current_logger()

args = {
    "preprocess_task_id": "ecb151123eda4c039a8df601d5d121b5",   # fill with Step 2 task id
    "force_cpu": False,
    "hidden_size": 128,
    "num_layers": 2,
    "dropout": 0.3,
    "batch_size": 64,
    "learning_rate": 0.001,
    "epochs": 20,
    "random_seed": 42,
}
task.connect(args)

# task.execute_remotely()

def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def get_device(force_cpu=False):
    if force_cpu:
        return torch.device("cpu")
    if torch.cuda.is_available():
        return torch.device("cuda")
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")

class TimeSeriesDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.float32)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

class MultiRegionLSTM(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size=1, dropout=0.2):
        super(MultiRegionLSTM, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        effective_dropout = dropout if num_layers > 1 else 0.0
        self.lstm = nn.LSTM(
            input_size,
            hidden_size,
            num_layers,
            batch_first=True,
            dropout=effective_dropout
        )
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size, device=x.device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size, device=x.device)
        out, _ = self.lstm(x, (h0, c0))
        out = self.fc(out[:, -1, :])
        return out

set_seed(args["random_seed"])
device = get_device(args["force_cpu"])
print("Using device:", device)

if not args["preprocess_task_id"]:
    raise ValueError("Missing preprocess_task_id")

print("Retrieving processed dataset artifacts...")
dataset_task = Task.get_task(task_id=args["preprocess_task_id"])

X_train = dataset_task.artifacts["X_train"].get()
X_test = dataset_task.artifacts["X_test"].get()
y_train = dataset_task.artifacts["y_train"].get()
y_test = dataset_task.artifacts["y_test"].get()
feature_names = dataset_task.artifacts["feature_names"].get()
target_region = dataset_task.artifacts["target_region"].get()

scaler_file = dataset_task.artifacts["scaler"].get_local_copy()
with open(scaler_file, "rb") as f:
    scaler = pickle.load(f)

print("Processed dataset loaded")
print("X_train shape:", X_train.shape)
print("X_test shape:", X_test.shape)
print("Target region:", target_region)

train_loader = DataLoader(
    TimeSeriesDataset(X_train, y_train),
    batch_size=args["batch_size"],
    shuffle=False
)
test_loader = DataLoader(
    TimeSeriesDataset(X_test, y_test),
    batch_size=args["batch_size"],
    shuffle=False
)

input_size = X_train.shape[2]
model = MultiRegionLSTM(
    input_size=input_size,
    hidden_size=args["hidden_size"],
    num_layers=args["num_layers"],
    dropout=args["dropout"]
).to(device)

criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=args["learning_rate"])

print(f"Model input features: {input_size}")

train_losses = []
for epoch in range(args["epochs"]):
    model.train()
    epoch_loss = 0.0
    n_batches = 0

    for batch_X, batch_y in train_loader:
        batch_X = batch_X.to(device)
        batch_y = batch_y.to(device)

        optimizer.zero_grad()
        outputs = model(batch_X).squeeze()
        loss = criterion(outputs, batch_y)
        loss.backward()
        optimizer.step()

        epoch_loss += loss.item()
        n_batches += 1

    avg_loss = epoch_loss / n_batches if n_batches > 0 else 0.0
    train_losses.append(avg_loss)
    train_rmse = np.sqrt(avg_loss)

    print(f"Epoch [{epoch + 1}/{args['epochs']}] Avg Loss: {avg_loss:.6f} | Train RMSE: {train_rmse:.4f}")
    logger.report_scalar("train", "loss", avg_loss, epoch)
    logger.report_scalar("train", "rmse", train_rmse, epoch)

model.eval()
preds_list, acts_list = [], []

with torch.no_grad():
    for batch_X, batch_y in test_loader:
        batch_X = batch_X.to(device)
        out = model(batch_X).cpu().numpy().reshape(-1)
        preds_list.append(out)
        acts_list.append(batch_y.numpy().reshape(-1))

preds = np.concatenate(preds_list)
acts = np.concatenate(acts_list)

n = len(preds)
n_features = X_train.shape[2]

dummy_pred = np.zeros((n, n_features))
dummy_act = np.zeros((n, n_features))
dummy_pred[:, 0] = preds
dummy_act[:, 0] = acts

inv_pred = scaler.inverse_transform(dummy_pred)[:, 0]
inv_act = scaler.inverse_transform(dummy_act)[:, 0]

mae = mean_absolute_error(inv_act, inv_pred)
rmse = np.sqrt(mean_squared_error(inv_act, inv_pred))
r2 = r2_score(inv_act, inv_pred)

nonzero = inv_act != 0
mape = (
    np.mean(np.abs((inv_act[nonzero] - inv_pred[nonzero]) / inv_act[nonzero])) * 100
    if nonzero.sum() > 0 else np.nan
)

print("\nFINAL RESULTS")
print(f"RMSE: {rmse:.2f} MW")
print(f"MAE : {mae:.2f} MW")
print(f"MAPE: {mape if not np.isnan(mape) else 'N/A'}%")
print(f"R2  : {r2:.4f}")

logger.report_scalar("evaluation", "RMSE", rmse, 0)
logger.report_scalar("evaluation", "MAE", mae, 0)
if not np.isnan(mape):
    logger.report_scalar("evaluation", "MAPE", mape, 0)
logger.report_scalar("evaluation", "R2", r2, 0)

os.makedirs("model_artifacts", exist_ok=True)
model_path = os.path.join("model_artifacts", "multi_region_lstm.pth")
torch.save(model.state_dict(), model_path)
task.upload_artifact("trained_model", artifact_object=model_path)

limit = min(500, len(inv_act))
plt.figure(figsize=(14, 5))
plt.plot(inv_act[:limit], label="Actual")
plt.plot(inv_pred[:limit], label="Predicted", linestyle="--")
plt.title(f"{target_region} Load Forecast - First {limit} Test Hours")
plt.xlabel("Hours")
plt.ylabel("Load (MW)")
plt.legend()
plt.grid(alpha=0.25)
plt.tight_layout()

plot_path = "power_load_prediction_plot.png"
plt.savefig(plot_path)
task.upload_artifact("prediction_plot", artifact_object=plot_path)

print("Task 3 completed.")