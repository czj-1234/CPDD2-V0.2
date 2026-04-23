import os
import random
import pickle
import numpy as np
import pandas as pd

from clearml import Task, Dataset
from sklearn.preprocessing import MinMaxScaler

task = Task.init(
    project_name="AI_Studio_Basic_Demo",
    task_name="Pipeline step 2 process dataset"
)

args = {
    "dataset_id": "bbb0063d31824eb9a1d7c130fd6d0369",   # fill with Step 1 ClearML Dataset ID
    "target_region": "PJME",
    "seq_length": 24,
    "test_ratio": 0.1,
    "random_seed": 42,
    "data_files": [
        "AEP_hourly.csv",
        "COMED_hourly.csv",
        "DAYTON_hourly.csv",
        "DEOK_hourly.csv",
        "DOM_hourly.csv",
        "DUQ_hourly.csv",
        "EKPC_hourly.csv",
        "FE_hourly.csv",
        "NI_hourly.csv",
        "PJME_hourly.csv",
        "PJMW_hourly.csv",
        "PJM_Load_hourly.csv",
        "pjm_hourly_est.csv"
    ]
}
task.connect(args)
print("Arguments:", args)

# task.execute_remotely()

def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)

def load_and_process_data(data_dir, data_files, target_region):
    print("Loading and merging multiple region files...")
    main_df = None
    loaded_regions = []

    for filename in data_files:
        path = os.path.join(data_dir, filename)
        if not os.path.exists(path):
            print(f"Warning: {path} not found. Skipping.")
            continue

        df = pd.read_csv(path)

        dt_col = None
        for candidate in ["Datetime", "datetime", "Date", "date", df.columns[0]]:
            if candidate in df.columns:
                dt_col = candidate
                break
        if dt_col is None:
            raise ValueError(f"No datetime-like column found in {path}")

        df = df.rename(columns={dt_col: "Datetime"})
        df["Datetime"] = pd.to_datetime(df["Datetime"])
        df = df.set_index("Datetime").sort_index()
        df = df[~df.index.duplicated(keep="first")]

        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        if not numeric_cols:
            print(f"Warning: no numeric columns found in {path}. Skipping.")
            continue

        col_name = next((c for c in numeric_cols if "MW" in c or "mw" in c), numeric_cols[0])

        region_name = os.path.basename(path).split("_")[0].upper()
        df = df.rename(columns={col_name: region_name})
        df = df[[region_name]]

        if main_df is None:
            main_df = df
        else:
            main_df = main_df.merge(df, left_index=True, right_index=True, how="outer")

        loaded_regions.append(region_name)

    if main_df is None or main_df.empty:
        raise ValueError("No data loaded. Check file paths and dataset availability.")

    main_df = main_df.sort_index()
    main_df["hour"] = main_df.index.hour
    main_df["day_of_week"] = main_df.index.dayofweek
    main_df["month"] = main_df.index.month
    main_df["day_of_year"] = main_df.index.dayofyear

    main_df = main_df.interpolate(method="time", limit_direction="both")
    main_df = main_df.ffill().bfill()

    cols = list(main_df.columns)
    target = target_region.upper()
    if target in cols:
        cols.insert(0, cols.pop(cols.index(target)))
        main_df = main_df[cols]
    else:
        raise ValueError(f"Target region {target} not found. Loaded regions: {loaded_regions}")

    print(f"Merged Data Shape: {main_df.shape}")
    print(f"Features: {list(main_df.columns)}")
    print(f"Time Range: {main_df.index.min()} to {main_df.index.max()}")
    return main_df

def create_sequences(data_values, seq_length):
    sequences = []
    labels = []
    target_idx = 0
    for i in range(len(data_values) - seq_length):
        seq = data_values[i:i + seq_length]
        label = data_values[i + seq_length, target_idx]
        sequences.append(seq)
        labels.append(label)
    return np.array(sequences), np.array(labels)

set_seed(args["random_seed"])

if not args["dataset_id"]:
    raise ValueError("Missing dataset_id")

dataset = Dataset.get(dataset_id=args["dataset_id"])
dataset_path = dataset.get_local_copy()
print("Dataset local path:", dataset_path)

df = load_and_process_data(
    data_dir=dataset_path,
    data_files=args["data_files"],
    target_region=args["target_region"]
)

scaler = MinMaxScaler()
scaled_data = scaler.fit_transform(df.values)

X, y = create_sequences(scaled_data, args["seq_length"])
print(f"Created sequences. X shape: {X.shape}, y shape: {y.shape}")

split_idx = int(len(X) * (1 - args["test_ratio"]))
X_train, X_test = X[:split_idx], X[split_idx:]
y_train, y_test = y[:split_idx], y[split_idx:]

print("Uploading processed dataset artifacts...")

task.upload_artifact("X_train", X_train)
task.upload_artifact("X_test", X_test)
task.upload_artifact("y_train", y_train)
task.upload_artifact("y_test", y_test)
task.upload_artifact("feature_names", list(df.columns))
task.upload_artifact("target_region", args["target_region"])
task.upload_artifact("seq_length", args["seq_length"])

scaler_path = "scaler.pkl"
with open(scaler_path, "wb") as f:
    pickle.dump(scaler, f)
task.upload_artifact("scaler", artifact_object=scaler_path)

print("Step 2 completed.")