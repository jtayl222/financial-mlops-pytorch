import mlflow
import os


# Get experiment info
experiment = mlflow.get_experiment('22')
print(f"Experiment: {experiment.name}")

# List runs
runs = mlflow.search_runs(experiment_ids=['22'])
print(f"\nFound {len(runs)} runs")

# Check for logged models
for idx, run in runs.iterrows():
    run_id = run['run_id']
    print(f"\nRun ID: {run_id}")
    print(f"Status: {run['status']}")
    print(f"Metrics: {run.filter(regex='^metrics\\.').to_dict()}")
    
    # Check if model was logged
    client = mlflow.tracking.MlflowClient()
    artifacts = client.list_artifacts(run_id)
    for artifact in artifacts:
        print(f"  Artifact: {artifact.path}")