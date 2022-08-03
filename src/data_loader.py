import glob

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset
from imutils import paths


class ImageFolder(Dataset):
    '''
    Image shape is (128, 128, 3)
    '''

    def __init__(self, folder_path):
        self.files = sorted(glob.glob('%s/*.*' % folder_path))

    def __getitem__(self, index):
        path = self.files[index % len(self.files)]
        img = np.array(Image.open(path))
        img = np.transpose(img, (2, 0, 1))
        img = torch.from_numpy(img).float() / 255.0

        return img, path

    def get_random(self):
        i = np.random.randint(0, len(self.files))
        return self[i]

    def __len__(self):
        return len(self.files)


class RandomLoader():
    def __init__(self, folder_path):
        self.dataset_path = folder_path
        self.img_paths = sorted(list(paths.list_images(self.dataset_path)))
        self.dataset_size = len(self.img_paths)
        self.begin = 0

    def fetch_data(self):
        while True:
            rand_idx = np.random.randint(0, self.dataset_size)

            img = np.array(Image.open(self.img_paths[rand_idx]))
            img = np.transpose(img, (2, 0, 1))
            img = torch.from_numpy(img).float().unsqueeze(0) / 255.0

            yield img, self.img_paths[rand_idx]

    def fetch_data_in_order(self, restart=False):
        if restart:
            self.begin = 0

        img = np.array(Image.open(self.img_paths[self.begin]))
        img = np.transpose(img, (2, 0, 1))
        img = torch.from_numpy(img).float().unsqueeze(0) / 255.0

        name = self.img_paths[self.begin]

        self.begin += 1

        return img, name


class RandomBatchLoader():
    def __init__(self, folder_path):
        self.dataset_path = folder_path
        self.img_paths = sorted(list(paths.list_images(self.dataset_path)))
        self.dataset_size = len(self.img_paths)
        self.begin = 0

    def fetch_data(self, batch_size):
        rand_range = np.arange(0, self.dataset_size)
        imgs = np.zeros(shape=(batch_size, 3, 128, 128))
        rand_idxs = np.random.choice(rand_range, size=batch_size)
        names = []

        for i in range(batch_size):
            img = np.array(Image.open(self.img_paths[rand_idxs[i]]))
            img = np.transpose(img, (2, 0, 1))
            imgs[i, :, :, :] = img
            names.append(self.img_paths[rand_idxs[i]])
        imgs = torch.from_numpy(imgs).float() / 255.0

        return imgs, names

    def fetch_data_in_order(self, batch_size, restart=False):
        if restart:
            self.begin = 0
        imgs = np.zeros(shape=(batch_size, 3, 128, 128))
        names = []

        for i in range(batch_size):
            index = i + self.begin
            img = np.array(Image.open(self.img_paths[index]))
            img = np.transpose(img, (2, 0, 1))
            imgs[i, :, :, :] = img
            names.append(self.img_paths[index])
        imgs = torch.from_numpy(imgs).float() / 255.0
        self.begin += batch_size

        return imgs, names
