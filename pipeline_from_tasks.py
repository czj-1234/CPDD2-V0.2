from clearml.automation import PipelineController


def pre_execute_callback_example(a_pipeline, a_node, current_param_override):
    print(
        "Cloning Task id={} with parameters: {}".format(
            a_node.base_task_id, current_param_override
        )
    )
    return True


def post_execute_callback_example(a_pipeline, a_node):
    print("Completed Task id={}".format(a_node.executed))
    return


def run_pipeline():
    pipe = PipelineController(
        name="AI_Studio_Power_Load_Pipeline",
        project="AI_Studio_Basic_Demo",
        version="0.0.1",
        add_pipeline_tags=False,
    )

    pipe.add_step(
        name="stage_data",
        base_task_project="AI_Studio_Basic_Demo",
        base_task_name="Pipeline step 1 dataset artefact",
        pre_execute_callback=pre_execute_callback_example,
        post_execute_callback=post_execute_callback_example,
    )

    pipe.add_step(
        name="stage_process",
        parents=["stage_data"],
        base_task_project="AI_Studio_Basic_Demo",
        base_task_name="Pipeline step 2 process dataset",
        parameter_override={
            "General/dataset_id": "bbb0063d31824eb9a1d7c130fd6d0369",
            "General/target_region": "PJME",
            "General/seq_length": 24,
            "General/test_ratio": 0.1,
            "General/random_seed": 42,
        },
        pre_execute_callback=pre_execute_callback_example,
        post_execute_callback=post_execute_callback_example,
    )

    pipe.add_step(
        name="stage_train",
        parents=["stage_process"],
        base_task_project="AI_Studio_Basic_Demo",
        base_task_name="Pipeline step 3 train model",
        parameter_override={
            "General/preprocess_task_id": "${stage_process.id}",
            "General/force_cpu": False,
            "General/hidden_size": 128,
            "General/num_layers": 2,
            "General/dropout": 0.3,
            "General/batch_size": 64,
            "General/learning_rate": 0.001,
            "General/epochs": 20,
            "General/random_seed": 42,
        },
        pre_execute_callback=pre_execute_callback_example,
        post_execute_callback=post_execute_callback_example,
    )

    pipe.start_locally(run_pipeline_steps_locally=True)
    print("Pipeline completed locally and logged to ClearML. 🔥")


if __name__ == "__main__":
    run_pipeline()