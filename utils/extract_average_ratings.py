import torch
from tqdm import tqdm


def extract_embeddings(dataset, processor, model, device, timepoints):
    embeddings_dict = {}

    for idx in tqdm(range(len(dataset)), desc="Extracting embeddings"):
        key, images = dataset[idx]
        embeddings_for_case = {}

        for tp in timepoints:
            if tp in images:
                embeddings_for_case[tp] = []

                for image in images[tp]:
                    inputs = processor(images=image, return_tensors="pt")
                    pixel_values = inputs["pixel_values"].to(device)

                    with torch.no_grad():
                        outputs = model(pixel_values=pixel_values)
                        embedding = outputs.pooler_output

                    embeddings_for_case[tp].append(embedding.cpu().numpy())

        embeddings_dict[key] = embeddings_for_case

    return embeddings_dict
