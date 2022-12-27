from glob import glob
from torch.utils.data import Dataset
import numpy as np
import os
import torch


class ModelNet40(Dataset):
    def __init__(self, dataroot, transforms=None):
        super(ModelNet40, self).__init__()
        self.dataroot = dataroot
        self.transforms = transforms
        self.train_set = list(glob(os.path.join(self.dataroot, 'train', '*.npz')))
        # self.train_set = list(glob(os.path.join(self.dataroot, 'test', '*.npz')))
        self.test_set = list(glob(os.path.join(self.dataroot, 'test', '*.npz')))
        self.training = True
        self.pcd = self.train_set

        self.LABEL_DICT = {'airplane': 0, 'bathtub': 1, 'bed': 2, 'bench': 3, 'bookshelf': 4, 'bottle': 5, 'bowl': 6,
                           'car': 7, 'chair': 8, 'cone': 9, 'cup': 10, 'curtain': 11, 'desk': 12, 'door': 13,
                           'dresser': 14, 'flower': 15, 'glass': 16, 'guitar': 17, 'keyboard': 18, 'lamp': 19,
                           'laptop': 20, 'mantel': 21, 'monitor': 22, 'night': 23, 'person': 24, 'piano': 25,
                           'plant': 26, 'radio': 27, 'range': 28, 'sink': 29, 'sofa': 30, 'stairs': 31, 'stool': 32,
                           'table': 33, 'tent': 34, 'toilet': 35, 'tv': 36, 'vase': 37, 'wardrobe': 38, 'xbox': 39}
        # LABEL = set()
        # for x in self.test_set:
        #     LABEL.add(os.path.split(x)[1].split('_')[0])
        # LABEL = sorted(list(LABEL))
        # LABEL_DICT = {k: v for v, k in enumerate(LABEL)}

        print(
            f'Load ModelNet40 done, load {len(self.train_set)} items for training and {len(self.test_set)} items for testing.'
        )

    def __len__(self):
        return len(self.pcd)

    def __getitem__(self, index):
        # index = 0
        x = self.pcd[index]
        label = os.path.split(x)[1].split('_')[0]
        label = self.LABEL_DICT[label]
        with np.load(x) as npz:
            pcd, norm = npz['pcd'], npz['norm']
            x = torch.from_numpy(np.concatenate([pcd, norm], axis=1)).float()
        if self.transforms is not None:
            x = self.transforms(x)
        return x, torch.tensor([label], dtype=torch.long)

    def train(self):
        self.training = True
        self.pcd = self.train_set
        self.transforms.set_mode('train')

    def eval(self):
        self.training = False
        self.pcd = self.test_set
        self.transforms.set_mode('eval')
