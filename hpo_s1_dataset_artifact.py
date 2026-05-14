# "hpo_s1_dataset_artifact.py"

from clearml import Task, Dataset
import logging
from pathlib import Path

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

PROJECT_NAME = "AI_Studio_HPO_Demo"
TASK_NAME = "HPO step 1 dataset artifact"
DATASET_NAME = "Power Load Raw Dataset"
DATASET_PROJECT = "AI_Studio_HPO_Demo"
DATA_DIR = Path("work_dataset")


# Initialize the task
task = Task.init(
    project_name=PROJECT_NAME,
    task_name=TASK_NAME
)

# Only create the task, we will actually execute it later
#task.execute_remotely()


if not DATA_DIR.exists():
    raise FileNotFoundError(
        f"Cannot find dataset folder: {DATA_DIR}. "
        "Please make sure work_dataset is in the project root."
    )

csv_files = list(DATA_DIR.glob("*.csv"))

if len(csv_files) == 0:
    raise FileNotFoundError(
        f"No CSV files found in {DATA_DIR}. "
        "Please put your power load CSV files inside work_dataset."
    )

logger.info(f"Found {len(csv_files)} CSV files in {DATA_DIR}")
for file in csv_files:
    logger.info(f"Found file: {file.name}")


# Create a new dataset in ClearML
dataset = Dataset.create(
    dataset_name=DATASET_NAME,
    dataset_project=DATASET_PROJECT
)

# Add the whole work_dataset folder to the dataset
dataset.add_files(path=str(DATA_DIR))
logger.info(f"Added dataset folder: {DATA_DIR}")

# Upload the dataset
dataset.upload()
logger.info("Uploaded dataset files")

# Finalize the dataset
dataset.finalize()
logger.info(f"Dataset created with ID: {dataset.id}")

# Store the dataset ID as a task parameter for other steps to use
task.set_parameter("General/dataset_id", dataset.id)
logger.info(f"Stored dataset ID: {dataset.id}")

print("Power load dataset created and uploaded to ClearML")
print(f"Dataset ID: {dataset.id}")