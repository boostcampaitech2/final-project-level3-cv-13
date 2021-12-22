import cv2
import numpy as np
import os
import torch
from torch.utils.data import Dataset


class DetectionImageDataset(Dataset):
    """
    Dataset for Webtoon Face Detector
    """
    def __init__(self, data_path, select_names):
        super(DetectionImageDataset, self).__init__()
        self.imgs, self.paths, self.labels, self.idx_to_class = self._make_dataset(data_path, select_names)
        
        
    def __len__(self):
        return len(self.imgs)

    def __getitem__(self, idx):
        img = self.imgs[idx]
        path = self.paths[idx]
        label = self.labels[idx]
        return img, label, path

    def _make_dataset(self, data_path, select_names):
        imgs = []
        paths = []
        labels = []
        idx_to_class = {key:name for key, name in enumerate(select_names)}
        for key, name in idx_to_class.items():
            actor_path = os.path.join(data_path, name)
            for img_name in os.listdir(actor_path):
                # img의 차원이 3이 아니거나, 채널의 개수가 3이 아닌 경우 제외
                # 이미지 형식의 문제로 인해 이미지 로드가 불가능한 경우도 제외
                img_path = os.path.join(actor_path, img_name)
                img = cv2.imread(img_path)
                if type(img) == np.ndarray and len(img.shape) == 3 and img.shape[2] == 3:
                    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                    imgs.append(img)
                    paths.append(img_path)
                    labels.append(key)

        return imgs, paths, labels, idx_to_class


class EmbeddingDataset(Dataset):
    def __init__(self, aligned: list, names: list, img_names: list):
        aligned_torch = torch.stack(aligned)
        self.aligned_torch = aligned_torch
        self.names = names # actor name
        self.img_names = img_names

    def __len__(self):
        return len(self.aligned_torch)

    def __getitem__(self, idx):
        return self.aligned_torch[idx], self.names[idx], self.img_names[idx]