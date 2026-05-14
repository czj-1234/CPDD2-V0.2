from clearml.automation import PipelineController
import logging

# ============================================================
# Logging
# ============================================================

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Queue configuration
EXECUTION_QUEUE = "hpo_demo"

PROJECT_NAME = "AI_Studio_HPO_Demo"
PIPELINE_NAME = "AI_Studio_Power_Load_HPO_Pipeline"
PIPELINE_VERSION = "0.0.2"


def run_pipeline():
    # Connect ClearML with the current pipeline
    pipe = PipelineController(
        name=PIPELINE_NAME,
        project=PROJECT_NAME,
        version=PIPELINE_VERSION,
        add_pipeline_tags=False,
    )

    # Set default queue for pipeline control
    pipe.set_default_execution_queue(EXECUTION_QUEUE)
    logger.info(f"Set default execution queue to: {EXECUTION_QUEUE}")

    # ========================================================
    # Step 1: Upload raw power load dataset
    # ========================================================
    pipe.add_step(
        name="stage_data",
        base_task_project=PROJECT_NAME,
        base_task_name="HPO step 1 dataset artifact",
        execution_queue=EXECUTION_QUEUE,
    )

    # ========================================================
    # Step 2: Process power load dataset
    # ========================================================
    pipe.add_step(
        name="stage_process",
        parents=["stage_data"],
        base_task_project=PROJECT_NAME,
        base_task_name="HPO step 2 process dataset",
        execution_queue=EXECUTION_QUEUE,
        parameter_override={
            # Step 2 can read the latest dataset automatically,
            # but this parameter is kept for pipeline traceability.
            "General/dataset_id": "${stage_data.parameters.General/dataset_id}",
        },
    )

    # ========================================================
    # Step 3: Create / run base training task
    # ========================================================
    pipe.add_step(
        name="stage_train",
        parents=["stage_process"],
        base_task_project=PROJECT_NAME,
        base_task_name="HPO step 3 train model",
        execution_queue=EXECUTION_QUEUE,
        parameter_override={
            # hpo_s3_train_model.py can find Step 2 by task name,
            # but processed_task_id is passed explicitly here.
            "General/processed_task_id": "${stage_process.id}",
            "General/test_queue": EXECUTION_QUEUE,

            # Default training configuration
            "General/model_name": "lstm",
            "General/num_epochs": 20,
            "General/batch_size": 32,
            "General/learning_rate": 1e-3,
            "General/weight_decay": 1e-5,

            # Default model parameters
            "General/hidden_size": 64,
            "General/num_layers": 1,
            "General/dropout": 0.1,
        },
    )

    # ========================================================
    # Step 4: Hyperparameter optimization
    # ========================================================
    pipe.add_step(
        name="stage_hpo",
        parents=["stage_train", "stage_process"],
        base_task_project=PROJECT_NAME,
        base_task_name="HPO step 4 hyperparameter optimization",
        execution_queue=EXECUTION_QUEUE,
        parameter_override={
            # HPO uses stage_train as the base task.
            # The HPO script itself also finds the base task by name,
            # so this is mainly for record keeping.
            "General/base_train_task_id": "${stage_train.id}",
            "General/processed_task_id": "${stage_process.id}",
            "General/test_queue": EXECUTION_QUEUE,
        },
    )

    # ========================================================
    # Step 5: Final model training
    # ========================================================
    pipe.add_step(
        name="stage_final_model",
        parents=["stage_hpo", "stage_process"],
        base_task_project=PROJECT_NAME,
        base_task_name="HPO step 5 final model",
        execution_queue=EXECUTION_QUEUE,
        parameter_override={
            "General/processed_task_id": "${stage_process.id}",
            "General/best_task_id": "${stage_hpo.parameters.General/best_task_id}",
            "General/test_queue": EXECUTION_QUEUE,
        },
    )

    # Start the pipeline locally, while tasks run on the queue
    logger.info("Starting HPO pipeline locally with tasks on queue: %s", EXECUTION_QUEUE)
    pipe.start_locally(run_pipeline_steps_locally=True)
    logger.info("Pipeline started successfully")


if __name__ == "__main__":
    run_pipeline()