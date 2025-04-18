import os
from PIL import Image
from torch.utils.data import Dataset


class CustomImageDataset(Dataset):
    def __init__(self, img_dirs, prefixes, views, timepoints, transform=None):
        self.img_dirs = img_dirs
        self.prefixes = prefixes
        self.views = views
        self.timepoints = timepoints
        self.transform = transform
        self.samples = self._load_samples()

    def _load_samples(self):
        sample_dict = {}
        for img_dir, prefix in zip(self.img_dirs, self.prefixes):
            for root, _, files in os.walk(img_dir):
                for file in files:
                    if any(view in file for view in self.views):
                        tp = next((t for t in self.timepoints if t in file), None)
                        if tp:
                            key = f"{prefix}_{file.split('some_postfix')[0]}"
                            img_path = os.path.join(root, file)
                            if key not in sample_dict:
                                sample_dict[key] = {}
                            if tp not in sample_dict[key]:
                                sample_dict[key][tp] = []
                            sample_dict[key][tp].append(img_path)
        return sample_dict

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        key = list(self.samples.keys())[idx]
        timepoint_images = {}

        for tp in self.timepoints:
            if tp in self.samples[key]:
                images = []
                for img_path in self.samples[key][tp]:
                    image = Image.open(img_path).convert("RGB")
                    if self.transform:
                        image = self.transform(image)
                    images.append(image)
                timepoint_images[tp] = images

        return key, timepoint_images
