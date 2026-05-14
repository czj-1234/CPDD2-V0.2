# "hpo_s2_process_dataset.py"

from clearml import Task, Dataset
import logging
from pathlib import Path
import pickle

import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler


# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


PROJECT_NAME = "AI_Studio_HPO_Demo"
TASK_NAME = "HPO step 2 process dataset"

TARGET_REGION = "PJME"
SEQ_LENGTH = 24
TRAIN_RATIO = 0.8
VAL_RATIO = 0.1


REGION_FILES = {
    "AEP": "AEP_hourly.csv",
    "COMED": "COMED_hourly.csv",
    "DAYTON": "DAYTON_hourly.csv",
    "DEOK": "DEOK_hourly.csv",
    "DOM": "DOM_hourly.csv",
    "DUQ": "DUQ_hourly.csv",
    "EKPC": "EKPC_hourly.csv",
    "FE": "FE_hourly.csv",
    "NI": "NI_hourly.csv",
    "PJME": "PJME_hourly.csv",
    "PJMW": "PJMW_hourly.csv",
}


def detect_datetime_column(df):
    possible_cols = [
        "Datetime",
        "datetime",
        "Date",
        "date",
        "DATE",
        "timestamp",
        "Timestamp",
    ]

    for col in possible_cols:
        if col in df.columns:
            return col

    return df.columns[0]


def detect_value_column(df, region_name):
    if region_name in df.columns:
        return region_name

    possible_cols = [
        "value",
        "Value",
        "load",
        "Load",
        "MW",
        "mw",
        "PJM_Load",
        "PJME_MW",
        "PJMW_MW",
    ]

    for col in possible_cols:
        if col in df.columns:
            return col

    datetime_col = detect_datetime_column(df)
    candidate_cols = [c for c in df.columns if c != datetime_col]

    if len(candidate_cols) == 0:
        raise ValueError(f"Cannot detect value column for region {region_name}")

    return candidate_cols[0]


def load_region_file(file_path, region_name):
    df = pd.read_csv(file_path)

    datetime_col = detect_datetime_column(df)
    value_col = detect_value_column(df, region_name)

    df = df[[datetime_col, value_col]].copy()
    df.columns = ["Datetime", region_name]

    df["Datetime"] = pd.to_datetime(df["Datetime"], errors="coerce")
    df = df.dropna(subset=["Datetime"])
    df = df.sort_values("Datetime")
    df = df.drop_duplicates(subset=["Datetime"], keep="first")

    return df


def load_and_merge_power_data(data_dir):
    merged_df = None
    used_regions = []

    for region, filename in REGION_FILES.items():
        file_path = Path(data_dir) / filename

        if not file_path.exists():
            logger.warning(f"File not found, skipped: {file_path}")
            continue

        region_df = load_region_file(file_path, region)

        if merged_df is None:
            merged_df = region_df
        else:
            merged_df = pd.merge(
                merged_df,
                region_df,
                on="Datetime",
                how="outer",
            )

        used_regions.append(region)

    if merged_df is None:
        raise FileNotFoundError(
            f"No valid regional CSV files were found in {data_dir}"
        )

    if TARGET_REGION not in merged_df.columns:
        raise ValueError(
            f"Target region {TARGET_REGION} not found. "
            f"Available columns: {merged_df.columns.tolist()}"
        )

    merged_df = merged_df.sort_values("Datetime").reset_index(drop=True)

    numeric_cols = [c for c in merged_df.columns if c != "Datetime"]
    merged_df[numeric_cols] = merged_df[numeric_cols].interpolate(method="linear")
    merged_df[numeric_cols] = merged_df[numeric_cols].ffill().bfill()

    logger.info(f"Loaded regions: {used_regions}")
    logger.info(f"Merged dataframe shape: {merged_df.shape}")

    return merged_df


def add_time_features(df):
    df = df.copy()

    df["hour"] = df["Datetime"].dt.hour
    df["day_of_week"] = df["Datetime"].dt.dayofweek
    df["month"] = df["Datetime"].dt.month
    df["day_of_year"] = df["Datetime"].dt.dayofyear

    df["hour_sin"] = np.sin(2 * np.pi * df["hour"] / 24)
    df["hour_cos"] = np.cos(2 * np.pi * df["hour"] / 24)

    df["day_of_week_sin"] = np.sin(2 * np.pi * df["day_of_week"] / 7)
    df["day_of_week_cos"] = np.cos(2 * np.pi * df["day_of_week"] / 7)

    df["month_sin"] = np.sin(2 * np.pi * df["month"] / 12)
    df["month_cos"] = np.cos(2 * np.pi * df["month"] / 12)

    return df


def split_by_time(df, train_ratio=0.8, val_ratio=0.1):
    n = len(df)

    train_end = int(n * train_ratio)
    val_end = int(n * (train_ratio + val_ratio))

    train_df = df.iloc[:train_end].copy()
    val_df = df.iloc[train_end:val_end].copy()
    test_df = df.iloc[val_end:].copy()

    return train_df, val_df, test_df


def create_sequences(data_array, target_array, seq_length):
    X = []
    y = []

    for i in range(len(data_array) - seq_length):
        X.append(data_array[i : i + seq_length])
        y.append(target_array[i + seq_length])

    return np.array(X, dtype=np.float32), np.array(y, dtype=np.float32)


def save_pickle(obj, file_name):
    with open(file_name, "wb") as f:
        pickle.dump(obj, f)


# Initialize the task
task = Task.init(
    project_name=PROJECT_NAME,
    task_name=TASK_NAME,
)

# Only create the task, we will actually execute it later
#task.execute_remotely()


# Get dataset ID from Step 1
dataset_id = task.get_parameter("General/dataset_id")

if dataset_id is None:
    logger.warning(
        "No dataset_id found in task parameters. "
        "Trying to get the latest dataset named 'Power Load Raw Dataset'."
    )

    dataset = Dataset.get(
        dataset_project=PROJECT_NAME,
        dataset_name="Power Load Raw Dataset",
    )
else:
    dataset = Dataset.get(dataset_id=dataset_id)


# Download ClearML dataset
local_dataset_path = Path(dataset.get_local_copy())
logger.info(f"Dataset downloaded to: {local_dataset_path}")

# Sometimes ClearML stores the uploaded folder directly,
# and sometimes it stores files inside a work_dataset subfolder.
candidate_dirs = [
    local_dataset_path / "work_dataset",
    local_dataset_path,
]

data_dir = None
for candidate in candidate_dirs:
    if candidate.exists() and len(list(candidate.glob("*.csv"))) > 0:
        data_dir = candidate
        break

if data_dir is None:
    raise FileNotFoundError(
        f"Could not find CSV files in downloaded dataset: {local_dataset_path}"
    )

logger.info(f"Using data directory: {data_dir}")


# Load and process data
df = load_and_merge_power_data(data_dir)
df = add_time_features(df)

feature_cols = [c for c in df.columns if c != "Datetime"]
target_index = feature_cols.index(TARGET_REGION)

train_df, val_df, test_df = split_by_time(
    df,
    train_ratio=TRAIN_RATIO,
    val_ratio=VAL_RATIO,
)

logger.info(f"Train dataframe shape: {train_df.shape}")
logger.info(f"Validation dataframe shape: {val_df.shape}")
logger.info(f"Test dataframe shape: {test_df.shape}")

scaler = MinMaxScaler()

train_features = scaler.fit_transform(train_df[feature_cols])
val_features = scaler.transform(val_df[feature_cols])
test_features = scaler.transform(test_df[feature_cols])

train_target = train_features[:, target_index]
val_target = val_features[:, target_index]
test_target = test_features[:, target_index]

X_train, y_train = create_sequences(train_features, train_target, SEQ_LENGTH)
X_val, y_val = create_sequences(val_features, val_target, SEQ_LENGTH)
X_test, y_test = create_sequences(test_features, test_target, SEQ_LENGTH)

logger.info(f"X_train shape: {X_train.shape}, y_train shape: {y_train.shape}")
logger.info(f"X_val shape: {X_val.shape}, y_val shape: {y_val.shape}")
logger.info(f"X_test shape: {X_test.shape}, y_test shape: {y_test.shape}")


# Save processed artifacts locally
save_pickle(X_train, "X_train.pkl")
save_pickle(y_train, "y_train.pkl")
save_pickle(X_val, "X_val.pkl")
save_pickle(y_val, "y_val.pkl")
save_pickle(X_test, "X_test.pkl")
save_pickle(y_test, "y_test.pkl")
save_pickle(scaler, "scaler.pkl")
save_pickle(feature_cols, "feature_cols.pkl")

metadata = {
    "target_region": TARGET_REGION,
    "target_index": target_index,
    "seq_length": SEQ_LENGTH,
    "train_ratio": TRAIN_RATIO,
    "val_ratio": VAL_RATIO,
    "raw_shape": df.shape,
    "train_shape": train_df.shape,
    "val_shape": val_df.shape,
    "test_shape": test_df.shape,
}

save_pickle(metadata, "metadata.pkl")


# Upload artifacts to ClearML
task.upload_artifact("X_train", artifact_object="X_train.pkl")
task.upload_artifact("y_train", artifact_object="y_train.pkl")
task.upload_artifact("X_val", artifact_object="X_val.pkl")
task.upload_artifact("y_val", artifact_object="y_val.pkl")
task.upload_artifact("X_test", artifact_object="X_test.pkl")
task.upload_artifact("y_test", artifact_object="y_test.pkl")
task.upload_artifact("scaler", artifact_object="scaler.pkl")
task.upload_artifact("feature_cols", artifact_object="feature_cols.pkl")
task.upload_artifact("metadata", artifact_object="metadata.pkl")

# Store useful parameters
task.set_parameter("General/target_region", TARGET_REGION)
task.set_parameter("General/target_index", target_index)
task.set_parameter("General/seq_length", SEQ_LENGTH)
task.set_parameter("General/num_features", len(feature_cols))

logger.info("Processed dataset artifacts uploaded successfully.")

print("Dataset processed and uploaded to ClearML artifacts.")
print(f"Target region: {TARGET_REGION}")
print(f"Sequence length: {SEQ_LENGTH}")
print(f"Number of features: {len(feature_cols)}")
print(f"X_train: {X_train.shape}")
print(f"X_val: {X_val.shape}")
print(f"X_test: {X_test.shape}")