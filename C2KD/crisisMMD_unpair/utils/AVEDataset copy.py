import torch
import torchvision as tv
from torchvision import transforms
from torch.utils.data import Dataset

import os
import cv2
import pandas as pd
import numpy as np
from numpy import newaxis
from collections import defaultdict
import random



class RavvdessDataset(Dataset):
    def __init__(self, file, audio_dir, image_dir, mode='train'):
        self.df = pd.read_csv(file)
        self.audio_dir = audio_dir
        self.image_dir = image_dir
        self.image = []
        self.audio = []
        self.label = []
        self.mode = mode
        classes = []

        self.data_root = '/home/ducca/CMKD/C2KD/ravdess_unpair/data/ravvdess'

        self.train_csv = os.path.join(self.data_root, 'data_file', 'spa_dl.csv')
        self.val_csv = os.path.join(self.data_root, 'data_file', 'spa_val.csv')
        self.test_csv = os.path.join(self.data_root, 'data_file', 'spa_test.csv')

        if mode == 'train':
            csv_file = self.train_csv
        elif mode == 'test':
            csv_file = self.test_csv
        else:
            csv_file = self.val_csv

        with open(self.test_csv, 'r') as f1:
            files = f1.readlines()
            for item in files:
                item = item.strip().split(',')
                if item[-1] not in classes:
                    classes.append(item[-1])
        class_dict = {}
        for i, c in enumerate(classes):
            class_dict[c] = i

        
        with open(csv_file, 'r') as f2:
            files = f2.readlines()
            for item in files:
                item = item.strip().split(',')
                audio_path = os.path.join(self.audio_dir, item[0])
                visual_path = os.path.join(self.image_dir, item[1])
                
                self.image.append(visual_path)
                self.audio.append(audio_path)
                self.label.append(class_dict[item[-1]])

        self.indices_per_class = defaultdict(list)
        for idx, lbl in enumerate(self.label):
            self.indices_per_class[lbl].append(idx)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, index):
        image_idx = index
        target_label = self.label[image_idx]

        if self.mode == 'train':
            candidate_indices = self.indices_per_class[target_label]

            audio_idx = random.choice(candidate_indices)
        else:
            audio_idx = index

        aud_name = os.path.join(self.audio_dir, self.df.iloc[audio_idx, 0])
        audio = np.load(aud_name)
        # audio_wrk = torch.from_numpy(audio[newaxis, :, :]) without normalization, not preferred
        audio = self.aud_transform(audio[:, :, newaxis])
        audio[:, :, [0, 2]] = audio[:, :, [2, 0]]

        if self.mode == 'train':
            transform = transforms.Compose([
                transforms.RandomResizedCrop(224),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ])
        else:
            transform = transforms.Compose([
                transforms.Resize(size=(224, 224)),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ])

        img_name = os.path.join(self.image_dir, self.df.iloc[image_idx, 1])
        image = cv2.imread(img_name)
        image = transform(image)
        # print(image.shape)
        image[:, :, [0, 2]] = image[:, :, [2, 0]]

        sample = {'audio': audio, 'image': image, 'label': self.df.iloc[index, 2]}
        return sample

