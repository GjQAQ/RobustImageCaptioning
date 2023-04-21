import random
import json

import torch
from torchvision import transforms

from aoanet.dataloader import DataLoader
from .dataset import MSCOCO2014


class _DataloaderWrapper(DataLoader):
    def __init__(self, encoder, root_path: str, opt, device='cpu'):
        super().__init__(opt)
        self.encoder = encoder.to(device)
        self.preprocess = transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        self.device = device

        self.datasets = {}
        with open(opt.input_json) as jsn:
            self.images_info = json.load(jsn)['images']
        for split in ('train', 'val', 'test'):
            self.datasets[split] = MSCOCO2014(root_path, opt.input_label_h5, self.images_info, split)

    def __getitem__(self, index):
        split = self.images_info[index]['split']
        if split == 'restval':
            split = 'train'
        image, _, _ = self.datasets[split][index]
        image = self.preprocess(image).to(self.device)[None, ...]

        fc_feat, att_feat = self.encoder(image)
        att_feat = att_feat.reshape(-1, att_feat.shape[-1])
        fc_feat = fc_feat.squeeze()
        seq = self.get_captions(index, self.seq_per_img)
        return [fc_feat, att_feat, seq, index]


class DataloaderWrapper(DataLoader):
    def __init__(self, encoder, root_path: str, opt, device='cpu'):
        super().__init__(opt)
        self.device = device
        self.encoder = encoder.to(device)
        self.preprocess = transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]).to(device)

        self.datasets = {}
        with open(opt.input_json) as jsn:
            self.images_info = json.load(jsn)['images']
        for split in ('train', 'val', 'test'):
            self.datasets[split] = MSCOCO2014(root_path, opt.input_label_h5, self.images_info, split)

    def __getitem__(self, index):
        # return only index in self.images_info without features
        return [index]

    def get_batch(self, split, batch_size=None):
        # this method must be overriden for following reasons:
        # 1. It must load images and labels rather than features
        # 2. It must load in batch rather than one by one
        if batch_size is not None:
            raise ValueError(f'The parameter batch_size is banned')
        batch_size = self.batch_size

        wrapped = False
        indices = []
        splits = []
        infos = []
        label_batch = []
        gts = []
        # images = []
        fc_batch = []
        att_batch = []

        for i in range(batch_size):
            ix, tmp_wrapped = self._prefetch_process[split].get()
            if tmp_wrapped:
                wrapped = True
            splits.append(split)
            indices.append(ix)
            infos.append({
                'ix': ix,
                'id': self.images_info[ix]['id'],
                'file_path': self.images_info[ix].get('file_path', '')
            })
        for index, split in zip(indices, splits):
            image, padded_cap, ground_truth = self.datasets[split][index]
            # images.append(self.preprocess(image))
            fc, att = self.encoder(self.preprocess(image.to(self.device))[None, ...])
            fc_batch.append(fc)
            att_batch.append(att.reshape(-1, att.shape[-1]))
            label_batch.append(padded_cap.to(self.device))
            gts.append(ground_truth.to(self.device))
        # image_batch = torch.stack(images).to(device)
        # fc_batch, att_batch = self.encoder(image_batch)
        fc_batch = torch.stack(fc_batch)
        att_batch = torch.stack(att_batch)

        data = {
            'fc_feats': torch.repeat_interleave(fc_batch, self.seq_per_img, 0),
            'att_feats': torch.repeat_interleave(att_batch, self.seq_per_img, 0),
            'att_masks': None,
            'labels': torch.cat(label_batch).to(torch.int64),
            'gts': gts,
            'infos': infos,
            'bounds': {
                'it_pos_now': self.iterators[split],
                'it_max': len(self.split_ix[split]),
                'wrapped': wrapped
            }
        }

        nonzeros = torch.tensor(list(map(lambda x: (x != 0).sum() + 2, data['labels'])))
        mask_batch = torch.zeros(data['labels'].shape[0], self.seq_length + 2)
        for ix, row in enumerate(mask_batch):
            row[:nonzeros[ix]] = 1
        data['masks'] = mask_batch.to(self.device)

        return data
