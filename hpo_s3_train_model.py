# "hpo_s3_train_model.py"

import os
os.environ.pop("MPLBACKEND", None)

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from clearml import Task
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

import xgboost as xgb
import numpy as np
import pandas as pd
import pickle
import logging
from pathlib import Path
from tqdm import tqdm


# ============================================================
# Logging and folders
# ============================================================

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

os.makedirs("assets", exist_ok=True)
os.makedirs("figs", exist_ok=True)


# ============================================================
# ClearML task
# ============================================================

task = Task.init(
    project_name="AI_Studio_HPO_Demo",
    task_name="HPO step 3 train model",
    task_type=Task.TaskTypes.training,
    reuse_last_task_id=False,
)

args = {
    "processed_task_id": "",
    "test_queue": "hpo_demo",

    # Model selection
    "model_name": "lstm",  # linear, xgboost, random_forest, lstm, gru, transformer

    # Common training parameters
    "num_epochs": 20,
    "batch_size": 32,
    "learning_rate": 1e-3,
    "weight_decay": 1e-5,

    # RNN parameters
    "hidden_size": 64,
    "num_layers": 1,
    "dropout": 0.1,

    # Transformer parameters
    "d_model": 64,
    "nhead": 4,
    "num_encoder_layers": 2,
    "dim_feedforward": 128,

    # XGBoost parameters
    "xgb_n_estimators": 200,
    "xgb_max_depth": 5,
    "xgb_learning_rate": 0.05,

    # Random forest parameters
    "rf_n_estimators": 200,
    "rf_max_depth": 10,
}

args = task.connect(args)
logger.info(f"Connected parameters: {args}")

# Execute the task remotely
#task.execute_remotely()


# ============================================================
# Helper functions
# ============================================================

def load_pickle_from_artifact(source_task, artifact_name):
    artifact_path = source_task.artifacts[artifact_name].get_local_copy()
    with open(artifact_path, "rb") as f:
        return pickle.load(f)


def flatten_sequences(X):
    """
    Convert [N, T, F] to [N, T*F] for traditional ML models.
    """
    return X.reshape(X.shape[0], -1)


def inverse_transform_target(y_scaled, scaler, target_index, num_features):
    """
    Convert scaled target values back to original load scale.
    """
    y_scaled = np.asarray(y_scaled).reshape(-1)

    dummy = np.zeros((len(y_scaled), num_features))
    dummy[:, target_index] = y_scaled

    inversed = scaler.inverse_transform(dummy)
    return inversed[:, target_index]


def calculate_metrics(y_true, y_pred):
    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_true, y_pred)

    mape = np.mean(
        np.abs((y_true - y_pred) / np.clip(np.abs(y_true), 1e-8, None))
    ) * 100

    r2 = r2_score(y_true, y_pred)

    return {
        "MSE": float(mse),
        "RMSE": float(rmse),
        "MAE": float(mae),
        "MAPE": float(mape),
        "R2": float(r2),
    }


def plot_prediction(y_true, y_pred, save_path, title):
    plt.figure(figsize=(12, 5))
    plt.plot(y_true[:300], label="Actual")
    plt.plot(y_pred[:300], label="Predicted")
    plt.title(title)
    plt.xlabel("Time Step")
    plt.ylabel("Power Load")
    plt.legend()
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()


# ============================================================
# PyTorch models
# ============================================================

class LSTMForecaster(nn.Module):
    def __init__(self, input_size, hidden_size=64, num_layers=1, dropout=0.1):
        super(LSTMForecaster, self).__init__()

        lstm_dropout = dropout if num_layers > 1 else 0.0

        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=lstm_dropout,
        )

        self.fc = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, 1),
        )

    def forward(self, x):
        output, _ = self.lstm(x)
        last_output = output[:, -1, :]
        prediction = self.fc(last_output)
        return prediction.squeeze(-1)


class GRUForecaster(nn.Module):
    def __init__(self, input_size, hidden_size=64, num_layers=1, dropout=0.1):
        super(GRUForecaster, self).__init__()

        gru_dropout = dropout if num_layers > 1 else 0.0

        self.gru = nn.GRU(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=gru_dropout,
        )

        self.fc = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, 1),
        )

    def forward(self, x):
        output, _ = self.gru(x)
        last_output = output[:, -1, :]
        prediction = self.fc(last_output)
        return prediction.squeeze(-1)


class TransformerForecaster(nn.Module):
    def __init__(
        self,
        input_size,
        d_model=64,
        nhead=4,
        num_encoder_layers=2,
        dim_feedforward=128,
        dropout=0.1,
    ):
        super(TransformerForecaster, self).__init__()

        self.input_projection = nn.Linear(input_size, d_model)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,
        )

        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers=num_encoder_layers,
        )

        self.fc = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, 1),
        )

    def forward(self, x):
        x = self.input_projection(x)
        encoded = self.transformer_encoder(x)
        last_output = encoded[:, -1, :]
        prediction = self.fc(last_output)
        return prediction.squeeze(-1)


# ============================================================
# Training functions
# ============================================================

def train_torch_model(
    model,
    train_loader,
    val_loader,
    num_epochs,
    learning_rate,
    weight_decay,
    device,
):
    criterion = nn.MSELoss()

    optimizer = optim.Adam(
        model.parameters(),
        lr=learning_rate,
        weight_decay=weight_decay,
    )

    model.to(device)

    best_val_rmse = float("inf")
    best_state = None

    for epoch in tqdm(range(num_epochs), desc="Training Epochs"):
        model.train()
        train_losses = []

        for batch_X, batch_y in train_loader:
            batch_X = batch_X.to(device)
            batch_y = batch_y.to(device)

            optimizer.zero_grad()
            outputs = model(batch_X)
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()

            train_losses.append(loss.item())

        avg_train_loss = float(np.mean(train_losses))

        model.eval()
        val_preds = []
        val_targets = []

        with torch.no_grad():
            for batch_X, batch_y in val_loader:
                batch_X = batch_X.to(device)
                outputs = model(batch_X)

                val_preds.extend(outputs.cpu().numpy())
                val_targets.extend(batch_y.numpy())

        val_preds = np.array(val_preds)
        val_targets = np.array(val_targets)

        val_rmse_scaled = float(np.sqrt(mean_squared_error(val_targets, val_preds)))
        val_mae_scaled = float(mean_absolute_error(val_targets, val_preds))

        task.get_logger().report_scalar(
            title="train",
            series="loss",
            value=avg_train_loss,
            iteration=epoch,
        )

        task.get_logger().report_scalar(
            title="validation",
            series="RMSE",
            value=val_rmse_scaled,
            iteration=epoch,
        )

        task.get_logger().report_scalar(
            title="validation",
            series="MAE",
            value=val_mae_scaled,
            iteration=epoch,
        )

        if val_rmse_scaled < best_val_rmse:
            best_val_rmse = val_rmse_scaled
            best_state = model.state_dict()

    if best_state is not None:
        model.load_state_dict(best_state)

    # Important for HPO objective
    task.get_logger().report_scalar(
        title="validation",
        series="RMSE",
        value=best_val_rmse,
        iteration=0,
    )

    return model, best_val_rmse


def predict_torch_model(model, loader, device):
    model.eval()
    preds = []
    targets = []

    with torch.no_grad():
        for batch_X, batch_y in loader:
            batch_X = batch_X.to(device)
            outputs = model(batch_X)

            preds.extend(outputs.cpu().numpy())
            targets.extend(batch_y.numpy())

    return np.array(targets), np.array(preds)


# ============================================================
# Load processed artifacts from Step 2
# ============================================================

processed_task_id = args.get("processed_task_id", "")

if not processed_task_id:
    processed_task_id = task.get_parameter("General/processed_task_id")

if not processed_task_id:
    logger.warning(
        "No processed_task_id found. Trying to find task by project and task name."
    )
    processed_task = Task.get_task(
        project_name="AI_Studio_HPO_Demo",
        task_name="HPO step 2 process dataset",
    )
else:
    processed_task = Task.get_task(task_id=processed_task_id)

logger.info(f"Using processed task ID: {processed_task.id}")

X_train = load_pickle_from_artifact(processed_task, "X_train")
y_train = load_pickle_from_artifact(processed_task, "y_train")
X_val = load_pickle_from_artifact(processed_task, "X_val")
y_val = load_pickle_from_artifact(processed_task, "y_val")
X_test = load_pickle_from_artifact(processed_task, "X_test")
y_test = load_pickle_from_artifact(processed_task, "y_test")
scaler = load_pickle_from_artifact(processed_task, "scaler")
feature_cols = load_pickle_from_artifact(processed_task, "feature_cols")
metadata = load_pickle_from_artifact(processed_task, "metadata")

target_index = metadata["target_index"]
target_region = metadata["target_region"]
num_features = len(feature_cols)

logger.info(f"X_train shape: {X_train.shape}")
logger.info(f"X_val shape: {X_val.shape}")
logger.info(f"X_test shape: {X_test.shape}")
logger.info(f"Target region: {target_region}")
logger.info(f"Number of features: {num_features}")


# ============================================================
# Select and train model
# ============================================================

model_name = args["model_name"].lower()
logger.info(f"Selected model: {model_name}")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
logger.info(f"Using device: {device}")

if model_name in ["linear", "xgboost", "random_forest"]:
    X_train_flat = flatten_sequences(X_train)
    X_val_flat = flatten_sequences(X_val)
    X_test_flat = flatten_sequences(X_test)

    if model_name == "linear":
        model = LinearRegression()
        model.fit(X_train_flat, y_train)

    elif model_name == "xgboost":
        model = xgb.XGBRegressor(
            n_estimators=int(args["xgb_n_estimators"]),
            max_depth=int(args["xgb_max_depth"]),
            learning_rate=float(args["xgb_learning_rate"]),
            objective="reg:squarederror",
            random_state=42,
            n_jobs=-1,
        )
        model.fit(
            X_train_flat,
            y_train,
            eval_set=[(X_val_flat, y_val)],
            verbose=False,
        )

    elif model_name == "random_forest":
        model = RandomForestRegressor(
            n_estimators=int(args["rf_n_estimators"]),
            max_depth=int(args["rf_max_depth"]),
            random_state=42,
            n_jobs=-1,
        )
        model.fit(X_train_flat, y_train)

    val_pred_scaled = model.predict(X_val_flat)
    test_pred_scaled = model.predict(X_test_flat)

    val_rmse_scaled = float(np.sqrt(mean_squared_error(y_val, val_pred_scaled)))

    # Important for HPO objective
    task.get_logger().report_scalar(
        title="validation",
        series="RMSE",
        value=val_rmse_scaled,
        iteration=0,
    )

    model_path = "model.pkl"
    with open(model_path, "wb") as f:
        pickle.dump(model, f)

else:
    train_dataset = TensorDataset(
        torch.FloatTensor(X_train),
        torch.FloatTensor(y_train),
    )
    val_dataset = TensorDataset(
        torch.FloatTensor(X_val),
        torch.FloatTensor(y_val),
    )
    test_dataset = TensorDataset(
        torch.FloatTensor(X_test),
        torch.FloatTensor(y_test),
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=int(args["batch_size"]),
        shuffle=True,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=int(args["batch_size"]),
        shuffle=False,
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=int(args["batch_size"]),
        shuffle=False,
    )

    input_size = X_train.shape[2]

    if model_name == "lstm":
        model = LSTMForecaster(
            input_size=input_size,
            hidden_size=int(args["hidden_size"]),
            num_layers=int(args["num_layers"]),
            dropout=float(args["dropout"]),
        )

    elif model_name == "gru":
        model = GRUForecaster(
            input_size=input_size,
            hidden_size=int(args["hidden_size"]),
            num_layers=int(args["num_layers"]),
            dropout=float(args["dropout"]),
        )

    elif model_name == "transformer":
        model = TransformerForecaster(
            input_size=input_size,
            d_model=int(args["d_model"]),
            nhead=int(args["nhead"]),
            num_encoder_layers=int(args["num_encoder_layers"]),
            dim_feedforward=int(args["dim_feedforward"]),
            dropout=float(args["dropout"]),
        )

    else:
        raise ValueError(
            f"Unsupported model_name: {model_name}. "
            "Please choose from linear, xgboost, random_forest, lstm, gru, transformer."
        )

    model, val_rmse_scaled = train_torch_model(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        num_epochs=int(args["num_epochs"]),
        learning_rate=float(args["learning_rate"]),
        weight_decay=float(args["weight_decay"]),
        device=device,
    )
    _, val_pred_scaled = predict_torch_model(model, val_loader, device)
    _, test_pred_scaled = predict_torch_model(model, test_loader, device)

    model_path = "model.pth"
    torch.save(
        {
            "model_name": model_name,
            "model_state_dict": model.state_dict(),
            "args": dict(args),
            "input_size": input_size,
        },
        model_path,
    )


# ============================================================
# Convert predictions back to original scale
# ============================================================

y_val_original = inverse_transform_target(
    y_val,
    scaler,
    target_index,
    num_features,
)

val_pred_original = inverse_transform_target(
    val_pred_scaled,
    scaler,
    target_index,
    num_features,
)

y_test_original = inverse_transform_target(
    y_test,
    scaler,
    target_index,
    num_features,
)

test_pred_original = inverse_transform_target(
    test_pred_scaled,
    scaler,
    target_index,
    num_features,
)

val_metrics = calculate_metrics(y_val_original, val_pred_original)
test_metrics = calculate_metrics(y_test_original, test_pred_original)

logger.info(f"Validation metrics: {val_metrics}")
logger.info(f"Test metrics: {test_metrics}")


# ============================================================
# Report metrics to ClearML
# ============================================================

for metric_name, metric_value in val_metrics.items():
    task.get_logger().report_scalar(
        title="Validation Original Scale",
        series=metric_name,
        value=metric_value,
        iteration=0,
    )

for metric_name, metric_value in test_metrics.items():
    task.get_logger().report_scalar(
        title="Test Original Scale",
        series=metric_name,
        value=metric_value,
        iteration=0,
    )

task.set_parameter("General/final_model_name", model_name)
task.set_parameter("General/validation_RMSE", val_metrics["RMSE"])
task.set_parameter("General/test_RMSE", test_metrics["RMSE"])
task.set_parameter("General/test_MAE", test_metrics["MAE"])
task.set_parameter("General/test_MAPE", test_metrics["MAPE"])
task.set_parameter("General/test_R2", test_metrics["R2"])


# ============================================================
# Save artifacts
# ============================================================

task.upload_artifact("model", model_path)

metrics_path = "metrics.pkl"
with open(metrics_path, "wb") as f:
    pickle.dump(
        {
            "model_name": model_name,
            "validation_metrics": val_metrics,
            "test_metrics": test_metrics,
            "args": dict(args),
        },
        f,
    )

task.upload_artifact("metrics", metrics_path)

prediction_path = "figs/prediction_plot.png"
plot_prediction(
    y_true=y_test_original,
    y_pred=test_pred_original,
    save_path=prediction_path,
    title=f"Power Load Forecasting - {model_name.upper()}",
)

task.upload_artifact("prediction_plot", prediction_path)

print("Training completed successfully")
print(f"Model name: {model_name}")
print(f"Validation RMSE: {val_metrics['RMSE']:.4f}")
print(f"Test RMSE: {test_metrics['RMSE']:.4f}")
print(f"Test MAE: {test_metrics['MAE']:.4f}")
print(f"Test MAPE: {test_metrics['MAPE']:.4f}")
print(f"Test R2: {test_metrics['R2']:.4f}")