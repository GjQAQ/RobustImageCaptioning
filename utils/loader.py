import random
import json

import torch
from torchvision import transforms

from aoanet.dataloader import DataLoader
from .dataset import MSCOCO2014


class DataloaderWrapper(DataLoader):
    def __init__(self, encoder, root_path: str, opt, device='cpu'):
        super().__init__(opt)
        self.device = device
        if encoder is not None:
            self.encoder = encoder.to(device)
        self.preprocess = transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]).to(device)

        self.datasets = {}
        self.split_ix['train'] = self.split_ix['train']
        with open(opt.input_json) as jsn:
            self.images_info = json.load(jsn)['images']
        for split in ('train', 'val', 'test'):
            self.datasets[split] = MSCOCO2014(root_path, opt.input_label_h5, self.images_info, split)

    def __getitem__(self, index):
        # return only index in self.images_info without features
        return [index]

    def get_batch(self, split, image_only=False):
        wrapped = False
        indices = []
        splits = []
        infos = []
        label_batch = []
        gts = []
        fc_batch = []
        att_batch = []
        images = []

        for i in range(self.batch_size):
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
            image = self.preprocess(image.to(self.device))
            if image_only:
                images.append(image)
            else:
                fc, att = self.encoder(image[None, ...])
                fc_batch.append(fc.squeeze())
                att_batch.append(att.reshape(-1, att.shape[-1]))
            label_batch.append(padded_cap.to(self.device))
            gts.append(ground_truth.to(self.device))

        data = {
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
        if image_only:
            data['images'] = images
        else:
            data['fc_feats'] = torch.repeat_interleave(torch.stack(fc_batch), self.seq_per_img, 0)
            data['att_feats'] = torch.repeat_interleave(torch.stack(att_batch), self.seq_per_img, 0)

        nonzeros = torch.tensor(list(map(lambda x: (x != 0).sum() + 2, data['labels'])))
        mask_batch = torch.zeros(data['labels'].shape[0], self.seq_length + 2)
        for ix, row in enumerate(mask_batch):
            row[:nonzeros[ix]] = 1
        data['masks'] = mask_batch.to(self.device)

        return data
