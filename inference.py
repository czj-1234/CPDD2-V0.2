import os
import pickle
import numpy as np
import pandas as pd
import torch
import torch.nn as nn


# =========================
# File paths
# =========================
MODEL_PATH = "final_model.pth"
SCALER_PATH = "scaler.pkl"
FEATURE_COLS_PATH = "feature_cols.pkl"

DEVICE = torch.device("cpu")
DEFAULT_SEQ_LENGTH = 24


class MultiRegionLSTM(nn.Module):
    def __init__(
        self,
        input_size,
        hidden_size,
        num_layers,
        fc_hidden_size=None,
        output_size=1,
        dropout=0.2
    ):
        super(MultiRegionLSTM, self).__init__()

        self.hidden_size = hidden_size
        self.num_layers = num_layers

        effective_dropout = dropout if num_layers > 1 else 0.0

        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=effective_dropout
        )

        # Match checkpoint style:
        # fc.0.weight, fc.0.bias, fc.3.weight, fc.3.bias
        if fc_hidden_size is not None:
            self.fc = nn.Sequential(
                nn.Linear(hidden_size, fc_hidden_size),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(fc_hidden_size, output_size)
            )
        else:
            self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        h0 = torch.zeros(
            self.num_layers,
            x.size(0),
            self.hidden_size,
            device=x.device
        )

        c0 = torch.zeros(
            self.num_layers,
            x.size(0),
            self.hidden_size,
            device=x.device
        )

        out, _ = self.lstm(x, (h0, c0))
        out = self.fc(out[:, -1, :])

        return out


def load_feature_cols():
    if not os.path.exists(FEATURE_COLS_PATH):
        raise FileNotFoundError(
            f"Feature columns file not found: {FEATURE_COLS_PATH}"
        )

    with open(FEATURE_COLS_PATH, "rb") as f:
        feature_cols = pickle.load(f)

    return list(feature_cols)


def load_scaler():
    if not os.path.exists(SCALER_PATH):
        raise FileNotFoundError(
            f"Scaler file not found: {SCALER_PATH}"
        )

    with open(SCALER_PATH, "rb") as f:
        scaler = pickle.load(f)

    return scaler


def get_state_dict_from_checkpoint(checkpoint):
    if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
        return checkpoint["model_state_dict"]

    return checkpoint


def infer_model_config(state_dict, fallback_input_size):
    """
    Infer model structure from checkpoint parameter shapes.
    """

    # Example:
    # lstm.weight_ih_l0 shape = [4 * hidden_size, input_size]
    lstm_w_ih_l0 = state_dict["lstm.weight_ih_l0"]

    hidden_size = lstm_w_ih_l0.shape[0] // 4
    input_size = lstm_w_ih_l0.shape[1]

    if input_size <= 0:
        input_size = fallback_input_size

    # Count LSTM layers
    num_layers = 0
    while f"lstm.weight_ih_l{num_layers}" in state_dict:
        num_layers += 1

    if num_layers == 0:
        num_layers = 1

    # Check whether fc is Sequential or Linear
    fc_hidden_size = None
    output_size = 1

    if "fc.0.weight" in state_dict and "fc.3.weight" in state_dict:
        fc_hidden_size = state_dict["fc.0.weight"].shape[0]
        output_size = state_dict["fc.3.weight"].shape[0]

    elif "fc.weight" in state_dict:
        output_size = state_dict["fc.weight"].shape[0]

    return {
        "input_size": input_size,
        "hidden_size": hidden_size,
        "num_layers": num_layers,
        "fc_hidden_size": fc_hidden_size,
        "output_size": output_size,
    }


def load_model(input_size):
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(
            f"Model file not found: {MODEL_PATH}"
        )

    checkpoint = torch.load(MODEL_PATH, map_location=DEVICE)
    state_dict = get_state_dict_from_checkpoint(checkpoint)

    config = infer_model_config(
        state_dict=state_dict,
        fallback_input_size=input_size
    )

    print("Loaded model config:", config)

    model = MultiRegionLSTM(
        input_size=config["input_size"],
        hidden_size=config["hidden_size"],
        num_layers=config["num_layers"],
        fc_hidden_size=config["fc_hidden_size"],
        output_size=config["output_size"],
        dropout=0.2
    ).to(DEVICE)

    model.load_state_dict(state_dict)
    model.eval()

    return model


def add_cyclical_features(user_inputs):
    user_inputs = dict(user_inputs)

    hour = user_inputs.get("hour", 0)
    day_of_week = user_inputs.get("day_of_week", 0)
    month = user_inputs.get("month", 1)

    user_inputs["hour_sin"] = user_inputs.get(
        "hour_sin",
        np.sin(2 * np.pi * hour / 24)
    )

    user_inputs["hour_cos"] = user_inputs.get(
        "hour_cos",
        np.cos(2 * np.pi * hour / 24)
    )

    user_inputs["day_of_week_sin"] = user_inputs.get(
        "day_of_week_sin",
        np.sin(2 * np.pi * day_of_week / 7)
    )

    user_inputs["day_of_week_cos"] = user_inputs.get(
        "day_of_week_cos",
        np.cos(2 * np.pi * day_of_week / 7)
    )

    user_inputs["month_sin"] = user_inputs.get(
        "month_sin",
        np.sin(2 * np.pi * month / 12)
    )

    user_inputs["month_cos"] = user_inputs.get(
        "month_cos",
        np.cos(2 * np.pi * month / 12)
    )

    return user_inputs


def build_input_sequence(user_inputs, scaler, feature_cols):
    user_inputs = add_cyclical_features(user_inputs)

    input_df = pd.DataFrame([user_inputs])

    # Remove old 17-feature GUI fields if they still exist
    input_df = input_df.drop(columns=["PJM_x", "PJM_y"], errors="ignore")

    # Align to the exact training feature columns
    input_df = input_df.reindex(columns=feature_cols, fill_value=0)

    print("Input feature shape before scaling:", input_df.shape)
    print("Feature columns:", list(input_df.columns))

    scaled_row = scaler.transform(input_df)[0]

    sequence = np.tile(scaled_row, (DEFAULT_SEQ_LENGTH, 1))
    sequence = np.expand_dims(sequence, axis=0)

    return sequence


def inverse_transform_target(pred_scaled, scaler, feature_cols):
    dummy = np.zeros((1, len(feature_cols)), dtype=np.float32)

    if "PJME" not in feature_cols:
        raise ValueError("PJME is not found in feature_cols.pkl")

    target_idx = feature_cols.index("PJME")
    dummy[0, target_idx] = pred_scaled

    inv = scaler.inverse_transform(dummy)[0, target_idx]

    return float(inv)


def predict_load(user_inputs):
    feature_cols = load_feature_cols()
    scaler = load_scaler()

    input_size = len(feature_cols)

    model = load_model(input_size=input_size)

    seq = build_input_sequence(
        user_inputs=user_inputs,
        scaler=scaler,
        feature_cols=feature_cols
    )

    seq_tensor = torch.tensor(seq, dtype=torch.float32).to(DEVICE)

    with torch.no_grad():
        pred_scaled = model(seq_tensor).cpu().numpy().reshape(-1)[0]

    pred_mw = inverse_transform_target(
        pred_scaled=pred_scaled,
        scaler=scaler,
        feature_cols=feature_cols
    )

    return pred_mw
