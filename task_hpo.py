# "task_hpo.py"

from clearml import Task
from clearml.automation import (
    HyperParameterOptimizer,
    DiscreteParameterRange,
    UniformParameterRange,
    RandomSearch,
)
import logging


# ============================================================
# Logging
# ============================================================

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# ============================================================
# ClearML HPO Task
# ============================================================

task = Task.init(
    project_name="AI_Studio_HPO_Demo",
    task_name="HPO step 4 hyperparameter optimization",
    task_type=Task.TaskTypes.optimizer,
    reuse_last_task_id=False,
)

# IMPORTANT:
# We want to run locally and log results to ClearML remote.
# Do NOT use task.execute_remotely().
# task.execute_remotely()


# ============================================================
# Get base training task
# ============================================================
# You must run hpo_s3_train_model.py once before running this file.
# This task will be cloned by ClearML HPO.

base_task = Task.get_task(
    project_name="AI_Studio_HPO_Demo",
    task_name="HPO step 3 train model",
)

if base_task is None:
    raise ValueError(
        "Could not find base task: 'HPO step 3 train model'. "
        "Please run `python hpo_s3_train_model.py` once before running task_hpo.py."
    )

logger.info(f"Base task ID: {base_task.id}")


# ============================================================
# Hyperparameter search space
# ============================================================
# model_name controls model selection.
#
# linear:
#   uses model_name only
#
# xgboost:
#   uses xgb_n_estimators, xgb_max_depth, xgb_learning_rate
#
# random_forest:
#   uses rf_n_estimators, rf_max_depth
#
# lstm / gru:
#   uses hidden_size, num_layers, dropout,
#        learning_rate, batch_size, num_epochs, weight_decay
#
# transformer:
#   uses d_model, nhead, num_encoder_layers,
#        dim_feedforward, dropout, learning_rate, batch_size

hyper_parameters = [
    # --------------------------------------------------------
    # Model selection
    # --------------------------------------------------------
    DiscreteParameterRange(
        "General/model_name",
        values=[
            "linear",
            "xgboost",
            "random_forest",
            "lstm",
            "gru",
            "transformer",
        ],
    ),

    # --------------------------------------------------------
    # Common deep learning training parameters
    # --------------------------------------------------------
    # Keep small for local CPU testing.
    DiscreteParameterRange(
        "General/num_epochs",
        values=[5, 10],
    ),

    DiscreteParameterRange(
        "General/batch_size",
        values=[32, 64],
    ),

    # Do NOT use LogUniformParameterRange here.
    # It produced unstable values such as learning_rate=1.0 before.
    DiscreteParameterRange(
        "General/learning_rate",
        values=[1e-4, 5e-4, 1e-3, 2e-3],
    ),

    DiscreteParameterRange(
        "General/weight_decay",
        values=[0.0, 1e-6, 1e-5, 1e-4],
    ),

    # --------------------------------------------------------
    # LSTM / GRU parameters
    # --------------------------------------------------------
    DiscreteParameterRange(
        "General/hidden_size",
        values=[32, 64, 128],
    ),

    DiscreteParameterRange(
        "General/num_layers",
        values=[1, 2],
    ),

    UniformParameterRange(
        "General/dropout",
        min_value=0.0,
        max_value=0.3,
        step_size=0.1,
    ),

    # --------------------------------------------------------
    # Transformer parameters
    # --------------------------------------------------------
    DiscreteParameterRange(
        "General/d_model",
        values=[32, 64, 128],
    ),

    DiscreteParameterRange(
        "General/nhead",
        values=[2, 4],
    ),

    DiscreteParameterRange(
        "General/num_encoder_layers",
        values=[1, 2],
    ),

    DiscreteParameterRange(
        "General/dim_feedforward",
        values=[64, 128, 256],
    ),

    # --------------------------------------------------------
    # XGBoost parameters
    # --------------------------------------------------------
    DiscreteParameterRange(
        "General/xgb_n_estimators",
        values=[100, 200, 300],
    ),

    DiscreteParameterRange(
        "General/xgb_max_depth",
        values=[3, 5, 7],
    ),

    DiscreteParameterRange(
        "General/xgb_learning_rate",
        values=[0.01, 0.03, 0.05, 0.1],
    ),

    # --------------------------------------------------------
    # Random Forest parameters
    # --------------------------------------------------------
    DiscreteParameterRange(
        "General/rf_n_estimators",
        values=[100, 200, 300],
    ),

    DiscreteParameterRange(
        "General/rf_max_depth",
        values=[5, 10, 20],
    ),
]


# ============================================================
# Hyperparameter optimizer
# ============================================================

optimizer = HyperParameterOptimizer(
    base_task_id=base_task.id,
    hyper_parameters=hyper_parameters,

    # hpo_s3_train_model.py must report:
    # title="validation", series="RMSE"
    objective_metric_title="validation",
    objective_metric_series="RMSE",
    objective_metric_sign="min",

    optimizer_class=RandomSearch,

    # IMPORTANT:
    # This queue is not used when we call start_locally().
    # It is kept here only for compatibility.
    execution_queue="default",

    # Local demo setting
    max_number_of_concurrent_tasks=1,
    total_max_jobs=4,

    # Stop each trial if it runs too long
    time_limit_per_job=60.0 * 60.0,

    # Overall HPO budget
    compute_time_limit=60.0 * 60.0 * 4.0,
)


# ============================================================
# Start HPO locally
# ============================================================
# IMPORTANT:
# Use start_locally() instead of start().
#
# optimizer.start() submits HPO trials to a ClearML queue.
# optimizer.start_locally() runs HPO trials on your local machine
# while still logging everything to ClearML remote.

optimizer.start_locally()
optimizer.wait()
optimizer.stop()


# ============================================================
# Get top experiments
# ============================================================

top_experiments = optimizer.get_top_experiments(top_k=3)

print("Top HPO experiments:")

for i, experiment in enumerate(top_experiments, start=1):
    print(f"Rank {i}:")
    print(f"Task ID: {experiment.id}")
    print(f"Task name: {experiment.name}")

    try:
        params = experiment.get_parameters()
        print("Selected model:", params.get("General/model_name"))
    except Exception as e:
        print(f"Could not read parameters: {e}")


# Store best task ID for final_model.py
if len(top_experiments) > 0:
    best_task = top_experiments[0]
    task.set_parameter("General/best_task_id", best_task.id)
    print(f"Best HPO task ID: {best_task.id}")
else:
    raise RuntimeError("No HPO experiments completed successfully.")


print("HPO completed successfully.")