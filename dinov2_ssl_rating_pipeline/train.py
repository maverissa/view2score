import os
from datetime import datetime
import hydra
import joblib
import numpy as np
import torch
from clearml import Task
from omegaconf import DictConfig
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split, KFold
from transformers import AutoImageProcessor, AutoModel
from dataset.custom_dataset import CustomImageDataset
from model.model import get_model
from utils.checkpoint_utils import remap_checkpoint, interpolate_pos_embed
from utils.embedding_utils import extract_embeddings
from utils.feature_utils import flatten_embeddings
from utils.rating_utils import extract_average_ratings


@hydra.main(config_path="config", config_name="config", version_base=None)
def main(cfg: DictConfig):
    task = Task.init(project_name="DINOv2Ratings", task_name=f"{cfg.experiment_id}_training", task_type=Task.TaskTypes.training)
    task.connect(cfg)
    exp_time = datetime.now().strftime("%Y%m%d_%H%M%S")
    exp_name = f"{cfg.experiment_id}_{exp_time}"
    output_dir = os.path.join(os.environ.get("OUTPUT_PATH", "."), exp_name)
    os.makedirs(output_dir, exist_ok=True)

    dataset = CustomImageDataset(cfg.data.img_dirs, cfg.data.prefixes, cfg.data.views, cfg.data.timepoints, transform=None)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    avg_ratings = extract_average_ratings(cfg.data.rating_file, cfg.data.prefixes, cfg.data.rating_columns)

    checkpoint = torch.load(cfg.model.teacher_checkpoint, map_location="cpu")
    state_dict = remap_checkpoint(checkpoint.get("teacher", checkpoint))

    processor = AutoImageProcessor.from_pretrained("facebook/dinov2-small")
    model = AutoModel.from_pretrained("facebook/dinov2-small")

    if "embeddings.position_embeddings" in state_dict:
        pos_embed_checkpoint = state_dict["embeddings.position_embeddings"]
        pos_embed_model = model.state_dict()["embeddings.position_embeddings"]
        if pos_embed_checkpoint.shape != pos_embed_model.shape:
            print("Interpolating positional embeddings...")
            state_dict["embeddings.position_embeddings"] = interpolate_pos_embed(pos_embed_checkpoint, pos_embed_model)

    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()

    embeddings_dict = extract_embeddings(
        dataset,
        processor,
        model,
        device,
        timepoints=cfg.data.timepoints
    )
    features, labels, keys = flatten_embeddings(
        embeddings_dict,
        avg_ratings,
        timepoints=cfg.data.timepoints,
        mode=cfg.model.feature_mode
    )

    X_train, X_test, y_train, y_test, key_train, key_test = train_test_split(features, labels, keys, test_size=0.2, random_state=cfg.model.random_state)

    final_model = get_model(cfg)
    final_model.fit(X_train, y_train)

    joblib.dump(final_model, os.path.join(output_dir, f"{cfg.model.type}_model.pkl"))
    np.savez(os.path.join(output_dir, "data.npz"), X_test=X_test, y_test=y_test, key_test=key_test)

    kf = KFold(n_splits=cfg.model.cv_folds, shuffle=True, random_state=cfg.model.random_state)
    for fold_number, (train_idx, val_idx) in enumerate(kf.split(features), 1):
        model_cv = get_model(cfg)
        model_cv.fit(features[train_idx], labels[train_idx])
        y_pred = model_cv.predict(features[val_idx])
        task.get_logger().report_scalar("CV Metrics", "MSE", value=mean_squared_error(labels[val_idx], y_pred), iteration=fold_number)
        task.get_logger().report_scalar("CV Metrics", "MAE", value=mean_absolute_error(labels[val_idx], y_pred), iteration=fold_number)
        task.get_logger().report_scalar("CV Metrics", "R2", value=r2_score(labels[val_idx], y_pred), iteration=fold_number)


if __name__ == "__main__":
    main()
