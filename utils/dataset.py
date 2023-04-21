import json
import os
import random
from typing import *

import numpy as np
import skimage.io
import torch
from torch.utils.data import Dataset
import h5py


class MSCOCO2014(Dataset):
    def __init__(
        self,
        root_path: str,
        h5_path: str,
        images_info: List[Dict],
        split='train',
        verbose=False
    ):
        if split not in ('train', 'val', 'test'):
            raise ValueError(f'Unknown split: {split}')
        self.split = split
        self.seq_per_img = 5

        # read images information
        self.root_path = root_path
        self.image_info = images_info

        # read labels information
        self.h5file = h5py.File(h5_path, 'r', driver='core')
        self.label = self.h5file['labels'][:]
        self.seq_length = self.h5file['labels'].shape[1]
        self.label_start = self.h5file['label_start_ix'][:]
        self.label_end = self.h5file['label_end_ix'][:]
        if verbose:
            print(f'Max sequence length in split {split} is: {self.seq_length}')

    def __len__(self):
        return len(self.imgs)

    def __getitem__(self, item):
        # read image
        img_path = os.path.join(self.root_path, self.image_info[item]['file_path'])
        image = skimage.io.imread(img_path)
        if len(image.shape) == 2:  # handle grayscale input images
            image = np.stack([image] * 3, axis=-1)
        image = image.transpose([2, 0, 1])
        image = image.astype('float32') / 255.0

        # read captions
        captions = self.get_captions(item)
        padded_cap = np.zeros([self.seq_per_img, self.seq_length + 2], dtype='int32')
        padded_cap[:, 1:self.seq_length + 1] = captions
        ground_truth = self.label[self.label_start[item] - 1:self.label_end[item]].astype(np.int32)

        return tuple(map(torch.from_numpy, (image, padded_cap, ground_truth)))

    def get_captions(self, index):
        """
        Couterpart for dataloader.Dataloader.get_captions in AoANet
        """
        start = self.label_start[index] - 1  # label_start starts from 1
        end = self.label_end[index] - 1
        num_of_captions = end - start + 1
        if num_of_captions <= 0:
            raise Warning(f'The image with index: {index} has no captions.')

        if num_of_captions < self.seq_per_img:  # we need to subsample (with replacement)
            seq = np.zeros([self.seq_per_img, self.seq_length], dtype='int')
            for q in range(self.seq_per_img):
                seq[q, :] = self.label[random.randint(start, end), :self.seq_length]
        else:
            offset = random.randint(start, end - self.seq_per_img + 1)
            seq = self.label[offset:offset + self.seq_per_img, :self.seq_length]
        return seq
