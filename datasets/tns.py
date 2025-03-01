import os
import json
import math
import numpy as np
from PIL import Image
from tqdm import tqdm

import torch
from torch.utils.data import Dataset, DataLoader, IterableDataset
import torchvision.transforms.functional as TF

import pytorch_lightning as pl

import datasets
from models.ray_utils import get_ray_directions
from utils.misc import get_rank


class BlenderTNSDatasetBase():
    def setup(self, config, split):
        self.config = config
        self.split = split
        self.rank = get_rank()

        self.has_mask = True
        self.apply_mask = self.config.apply_mask

        with open(os.path.join(self.config.root_dir, f"transforms_{self.split}.json"), 'r') as f:
            meta = json.load(f)

        if 'w' in meta and 'h' in meta:
            W, H = int(meta['w']), int(meta['h'])
        else:
            W, H = 1280, 720
        
        self.w, self.h = W, H
        self.img_wh = (self.w, self.h)

        #self.near, self.far = self.config.near_plane, self.config.far_plane

        self.focal = meta['fl_x']

        self.apply_mask = self.config.apply_mask

        # ray directions for all pixels, same for all images (same H, W, focal)
        self.directions = \
            get_ray_directions(self.w, self.h, meta['fl_x'], meta['fl_y'], meta['cx'], meta['cy']).to(self.rank) # (h, w, 3)           

        self.all_c2w, self.all_images, self.all_fg_masks, self.all_trans_masks = [], [], [], []

        for frame in tqdm(meta["frames"], desc=f"Loading {split}", unit="scene"):
            c2w = torch.from_numpy(np.array(frame['transform_matrix'], dtype=np.float32)[:3, :4])
            self.all_c2w.append(c2w)
            img_path = os.path.join(self.config.root_dir, f"{frame['file_path']}")
            img = Image.open(img_path)
            img = img.resize(self.img_wh, Image.BICUBIC)
            img = TF.to_tensor(img).permute(1, 2, 0) # (4, h, w) => (h, w, 4)

            mask_fg_path = os.path.join(self.config.root_dir, f"{frame['file_fg_segmap_path']}")
            mask_trans_path = os.path.join(self.config.root_dir, f"{frame['file_trans_segmap_path']}")

            self.all_trans_masks.append(torch.from_numpy(np.load(mask_trans_path)))
            self.all_fg_masks.append(torch.from_numpy(np.load(mask_fg_path)))
            self.all_images.append(img[...,:3])

        self.all_c2w, self.all_images, self.all_fg_masks, self.all_trans_masks = \
            torch.stack(self.all_c2w, dim=0).float().to(self.rank), \
            torch.stack(self.all_images, dim=0).float().to(self.rank), \
            torch.stack(self.all_fg_masks, dim=0).float().to(self.rank), \
            torch.stack(self.all_trans_masks, dim=0).float().to(self.rank)
        

class BlenderTNSDataset(Dataset, BlenderTNSDatasetBase):
    def __init__(self, config, split):
        self.setup(config, split)

    def __len__(self):
        return len(self.all_images)
    
    def __getitem__(self, index):
        return {
            'index': index
        }


class BlenderTNSIterableDataset(IterableDataset, BlenderTNSDatasetBase):
    def __init__(self, config, split):
        self.setup(config, split)

    def __iter__(self):
        while True:
            yield {}


@datasets.register('blender-tns')
class BlenderTNSDataModule(pl.LightningDataModule):
    def __init__(self, config):
        super().__init__()
        self.config = config
    
    def setup(self, stage=None):
        if stage in [None, 'fit']:
            self.train_dataset = BlenderTNSIterableDataset(self.config, self.config.train_split)
        if stage in [None, 'fit', 'validate']:
            self.val_dataset = BlenderTNSDataset(self.config, self.config.val_split)
        if stage in [None, 'test']:
            self.test_dataset = BlenderTNSDataset(self.config, self.config.test_split)
        if stage in [None, 'predict']:
            self.predict_dataset = BlenderTNSDataset(self.config, self.config.train_split)

    def prepare_data(self):
        pass
    
    def general_loader(self, dataset, batch_size):
        sampler = None
        return DataLoader(
            dataset, 
            num_workers=os.cpu_count(), 
            batch_size=batch_size,
            pin_memory=True,
            sampler=sampler
        )
    
    def train_dataloader(self):
        return self.general_loader(self.train_dataset, batch_size=1)

    def val_dataloader(self):
        return self.general_loader(self.val_dataset, batch_size=1)

    def test_dataloader(self):
        return self.general_loader(self.test_dataset, batch_size=1) 

    def predict_dataloader(self):
        return self.general_loader(self.predict_dataset, batch_size=1)       
