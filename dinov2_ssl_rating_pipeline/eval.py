import os
import hydra
import joblib
import numpy as np
from clearml import Task
from matplotlib import pyplot as plt
from omegaconf import DictConfig
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score


def run_evaluation(cfg: DictConfig):
    task = Task.init(project_name="DINOv2Ratings", task_name=f"{cfg.experiment_id}_evaluation", task_type=Task.TaskTypes.testing)
    base_dir = os.environ.get("OUTPUT_PATH", "outputs")
    matching_folders = [os.path.join(base_dir, d) for d in os.listdir(base_dir) if cfg.experiment_id in d]
    if not matching_folders:
        raise ValueError(f"No experiment found with ID: {cfg.experiment_id}")

    output_dir = sorted(matching_folders, key=os.path.getmtime)[-1]
    model = joblib.load(os.path.join(output_dir, "ridge_model.pkl"))
    data = np.load(os.path.join(output_dir, "data.npz"))
    X_test, y_test = data['X_test'], data['y_test']

    y_pred = model.predict(X_test)
    metrics = {
        "MSE": mean_squared_error(y_test, y_pred),
        "MAE": mean_absolute_error(y_test, y_pred),
        "R2": r2_score(y_test, y_pred)
    }
    for name, value in metrics.items():
        task.get_logger().report_scalar("Test Metrics", name, value=value, iteration=0)

    plt.figure(figsize=(8, 6))
    plt.scatter(y_test, y_pred, alpha=0.6)
    plt.plot([min(y_test), max(y_test)], [min(y_test), max(y_test)], 'r--')
    plot_path = os.path.join(output_dir, "test_evaluation.png")
    plt.savefig(plot_path)

    try:
        task.upload_artifact(name="evaluation_plot", artifact_object=plot_path)
    except Exception as e:
        task.get_logger().report_text(f"Artifact upload failed: {e}")

    return metrics


@hydra.main(config_path="config", config_name="config", version_base=None)
def evaluate_model(cfg: DictConfig):
    return run_evaluation(cfg)


if __name__ == "__main__":
    metrics = evaluate_model()
    if metrics:
        print("\nTest Set Evaluation:")
        for name, value in metrics.items():
            print(f"{name}: {value:.4f}")
    else:
        print("No metrics returned.")
