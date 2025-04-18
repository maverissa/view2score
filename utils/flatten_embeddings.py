import numpy as np


def flatten_embeddings(embeddings_dict, avg_ratings, timepoints, mode="concatenate"):
    features, labels, keys = [], [], []

    for key in embeddings_dict:
        if key not in avg_ratings:
            continue

        views = []
        for tp in timepoints:
            arr = np.array(embeddings_dict[key].get(tp, []))
            if arr.ndim == 1:
                arr = arr.reshape(1, -1)

            if mode == "average":
                views.append(arr.mean(axis=0))
            elif mode == "concatenate":
                views.append(arr.flatten())
            else:
                raise ValueError(f"Unknown feature mode: {mode}")

        features.append(np.concatenate(views))
        labels.append(avg_ratings[key])
        keys.append(key)

    return np.array(features), np.array(labels), np.array(keys)
