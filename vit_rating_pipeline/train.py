import os
from datetime import datetime

import hydra
import joblib
import numpy as np
import torch
import torch.nn as nn
from clearml import Task
from model.model import get_model
from omegaconf import DictConfig
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.model_selection import KFold, train_test_split
from torchvision import transforms
from tqdm import tqdm
from transformers import ViTForImageClassification

from dataset.custom_dataset import CustomImageDataset
from utils.rating_utils import extract_average_ratings


@hydra.main(config_path="config", config_name="config", version_base=None)
def main(cfg: DictConfig):
    task = Task.init(project_name="Project", task_name=f"{cfg.experiment_id}_training", task_type=Task.TaskTypes.training)
    task.connect(cfg)

    exp_time = datetime.now().strftime("%Y%m%d_%H%M%S")
    exp_name = f"{cfg.experiment_id}_{exp_time}"
    base_dir = os.environ.get("OUTPUT_PATH", ".")
    output_dir = os.path.join(base_dir, exp_name)
    os.makedirs(output_dir, exist_ok=True)

    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    dataset = CustomImageDataset(
        img_dirs=cfg.data.img_dirs,
        prefixes=cfg.data.prefixes,
        views=cfg.data.views,
        timepoints=cfg.data.timepoints,
        transform=transform
    )

    avg_ratings = extract_average_ratings(cfg.data.rating_file, cfg.data.prefixes, cfg.data.rating_columns)

    logits_cache_path = os.path.join(output_dir, "logits_dict.pkl")
    if cfg.model.get("use_cached_logits", False) and os.path.exists(logits_cache_path):
        logits_dict = joblib.load(logits_cache_path)
    else:
        model = ViTForImageClassification.from_pretrained(cfg.model.encoder)
        model.classifier = nn.Identity()
        model.eval()

        logits_dict = {}
        for idx in tqdm(range(len(dataset)), desc="Extracting embeddings"):
            key, images = dataset[idx]
            if key not in avg_ratings:
                continue
            logits_dict[key] = {}
            for tp in cfg.data.timepoints:
                if tp not in images:
                    logits_dict[key][tp] = [np.zeros(model.config.hidden_size)]
                    continue
                tp_logits = []
                for image in images[tp]:
                    image_batch = torch.stack([image])
                    with torch.no_grad():
                        tp_logits.append(model(image_batch).logits.squeeze().numpy())
                logits_dict[key][tp] = tp_logits
        joblib.dump(logits_dict, logits_cache_path)

    features, labels, keys = [], [], []
    for key in logits_dict:
        if key not in avg_ratings:
            continue
        tp_data = [np.array(logits_dict[key].get(tp, [np.zeros(model.config.hidden_size)])) for tp in cfg.data.timepoints]
        if cfg.model.feature_mode == "average":
            vec = np.concatenate([arr.mean(axis=0) for arr in tp_data])
        elif cfg.model.feature_mode == "concatenate":
            vec = np.concatenate([arr.flatten() for arr in tp_data])
        else:
            raise ValueError(f"Unsupported feature mode: {cfg.model.feature_mode}")
        features.append(vec)
        labels.append(avg_ratings[key])
        keys.append(key)

    features = np.array(features)
    labels = np.array(labels)
    keys = np.array(keys)

    X_train, X_test, y_train, y_test, key_train, key_test = train_test_split(
        features, labels, keys, test_size=0.2, random_state=cfg.model.random_state
    )

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
