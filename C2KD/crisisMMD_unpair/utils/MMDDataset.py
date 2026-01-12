import os
import pandas as pd
from PIL import Image

import torch
from torch.utils.data import Dataset
from torchvision import transforms
import random

class CrisisMMDDataset_paired(Dataset):
    def __init__(
        self,
        csv_file,
        image_root,
        tokenizer,
        max_length=128,
        image_transform=None,
        label_map=None
    ):
        """
        Args:
            csv_file: path to annotation CSV
            image_root: root folder containing images
            tokenizer: HuggingFace tokenizer
            max_length: max text length
            image_transform: torchvision transforms
            label_map: dict mapping string labels to int
        """
        self.df = pd.read_csv(csv_file)
        self.image_root = image_root
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.image_transform = image_transform

        self.label_map = label_map
        if self.label_map is not None:
            self.df["label"] = self.df["label"].map(label_map)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]

        # ---- text ----
        text = row["tweet_text"]
        encoding = self.tokenizer(
            text,
            padding="max_length",
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt"
        )

        # ---- image ----
        img_path = os.path.join(self.image_root, row["image_path"])
        image = Image.open(img_path).convert("RGB")

        if self.image_transform:
            image = self.image_transform(image)

        # ---- label ----
        label = torch.tensor(row["label"], dtype=torch.long)

        return {
            "input_ids": encoding["input_ids"].squeeze(0),
            "attention_mask": encoding["attention_mask"].squeeze(0),
            "image": image,
            "label": label
        }
    

class CrisisMMDDataset_unpaired(Dataset):
    def __init__(
        self,
        csv_file,
        image_root,
        tokenizer,
        max_length=128,
        image_transform=None,
        label_map=None
    ):
        self.df = pd.read_csv(csv_file)
        self.image_root = image_root
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.image_transform = image_transform

        if label_map is not None:
            self.df["label"] = self.df["label"].map(label_map)

        # ---- group indices by label ----
        self.label_to_indices = {}
        for idx, label in enumerate(self.df["label"]):
            self.label_to_indices.setdefault(label, []).append(idx)

        self.labels = list(self.label_to_indices.keys())

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        """
        idx selects the LABEL bucket, not a fixed pair
        """
        # ---- choose anchor row (for text) ----
        row_text = self.df.iloc[idx]
        label = row_text["label"]

        # ---- sample a DIFFERENT row with same label (for image) ----
        img_idx = random.choice(self.label_to_indices[label])
        row_img = self.df.iloc[img_idx]

        # =====================
        # Text
        # =====================
        encoding = self.tokenizer(
            row_text["tweet_text"],
            padding="max_length",
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt"
        )

        # =====================
        # Image
        # =====================
        img_path = os.path.join(self.image_root, row_img["image_path"])
        image = Image.open(img_path).convert("RGB")

        if self.image_transform:
            image = self.image_transform(image)

        return {
            "input_ids": encoding["input_ids"].squeeze(0),
            "attention_mask": encoding["attention_mask"].squeeze(0),
            "image": image,
            "label": torch.tensor(label, dtype=torch.long)
        }




