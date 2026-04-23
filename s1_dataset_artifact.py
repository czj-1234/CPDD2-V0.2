from clearml import Task, Dataset
import os

task = Task.init(
    project_name="AI_Studio_Basic_Demo",
    task_name="Pipeline step 1 dataset artefact"
)

# task.execute_remotely()

data_dir = os.path.join(os.path.dirname(__file__), "work_dataset")

if not os.path.exists(data_dir):
    raise FileNotFoundError(f"Data directory not found: {data_dir}")

dataset = Dataset.create(
    dataset_name="power_load_raw_data",
    dataset_project="AI_Studio_Basic_Demo"
)

dataset.add_files(path=data_dir)
dataset.upload()
dataset.finalize()

print("Raw dataset uploaded successfully.")
print(f"Data directory: {data_dir}")
print(f"ClearML Dataset ID: {dataset.id}")
print("Step 1 completed.")