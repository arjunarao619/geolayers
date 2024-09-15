import os
import argparse
import json
import torch

import random
import imageio
# the custom dataset file also includes scripts for geobench. if you dont want that, simply comment out those lines. 
from mmearth_dataset import get_mmearth_dataloaders
import numpy as np
from pathlib import Path

from geolayer_modalities.MODALITIES import BASELINE_MODALITIES_OUT, BASELINE_MODALITIES_IN, MODALITIES_FULL # this contains all the input and output bands u need for pretraining.

parser = argparse.ArgumentParser()
args = parser.parse_args()

# these 4 arguments need to be set manually
args.data_path = Path('/home/arjun/datasets/data_100k_v001') # path to h5 file 
args.random_crop = True # ensure that if the dataset image size is 128 x 128, the resulting image after cropping is 112 x 112.
args.random_crop_size = 112 # the size of the crop
args.batch_size = 1

# define the input and output bands for the dataset
args.inp_modalities = BASELINE_MODALITIES_IN
args.out_modalities = BASELINE_MODALITIES_OUT

args.modalities = args.inp_modalities.copy()
args.modalities.update(args.out_modalities) # args modalities is a dictionary of all the input and output bands.
args.modalities_full = MODALITIES_FULL # this is a dictionary of all the bands in the dataset.

args.no_ffcv = False # this flag allows you to load the ffcv dataloader or the h5 dataset.
args.processed_dir = None # default is automatically created in the data path. this is the dir where the beton file for ffcv is stored
args.num_workers = 4 # number of workers for the dataloader
args.distributed = False # if you are using distributed training, set this to True
def collate_fn(batch): # only for non ffcv dataloader
    # for each batch append the samples of the same modality together and return the ids. We keep track of the ids to differentiate between sentinel2_l1c and sentinel2_l2a
    return_batch = {}
    ids = [b['id'] for b in batch]
    return_batch = {modality: torch.stack([b[modality] for b in batch], dim=0) for modality in args.modalities.keys()}
    return ids, return_batch

train_dataloader = get_mmearth_dataloaders(
    args.data_path,
    args.processed_dir,
    args.modalities,
    splits = ["train", "val"],
    num_workers=args.num_workers,
    batch_size_per_device=args.batch_size,
    distributed=args.distributed,
)[0]

from IPython import embed;embed()

# Writing a naive plotting util to plot RGB imagery with their lat and lon as the title
# os.makedirs('/projects/arra4944/projects/geolayers/MMEarth-train/eda_plots', exist_ok=True)
# for i in range(80):
#     img = train_dataloader.__getitem__(random.randint(0, train_dataloader.__len__()))
#     rgb = img['sentinel2'][(3,2,1),:,:].T * 255 # Grab RGB bands
#     rgb = rgb.astype(np.uint8)
#     imageio.imwrite(os.path.join('/projects/arra4944/projects/geolayers/MMEarth-train/eda_plots',img['id'] + '.png'), rgb)





# Writing a naive plotting util to plot RGB imagery with their lat and lon as the title

