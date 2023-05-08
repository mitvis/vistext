# Visual feature extraction code for VisText image features.
# Usage: python vistext_proposal.py --data_dir ../../data/data --split test

import argparse
import cv2
from detectron2_proposal_maxnms import collate_fn, extract, NUM_OBJECTS, DIM
import json
import os
from pathlib import Path
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

class VisTextImageDataset(Dataset):
    """VisText dataset of chart images for feature extraction.
    
    Feature extraction code expects a dataset that returns a dictionary. The 
    dictionary must contain two keys: 'img_id' and 'img'. 'img_id' maps to the 
    unique image ID that is used to name the hdf5 dataset of the image features.
    'img' maps to the loaded open cv image of the chart. """

    def __init__(self, data_file, image_dir):
        """
        Args:
            data_file (string): Path to the pickle file with data samples.
            image_dir (string): Directory with all the images.
        """
        with open(data_file, 'r') as f:
            data = json.load(f)
        self.img_ids = list(set([datum['img_id'] for datum in data]))
        self.img_paths = [os.path.join(image_dir, f'{img_id}.png') for img_id in self.img_ids]

    def __len__(self):
        return len(self.img_ids)

    def __getitem__(self, idx):
        """
        Returns: a dictionary containing 'img_id' mapping to the image id from
            the data file and 'img' mapping to the open cv image of the chart.
        """
        img_id = self.img_ids[idx]
        img_path = self.img_paths[idx]
        img = cv2.imread(str(img_path))
        
        return {
            'img_id': img_id,
            'img': img
        }


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--data_dir', 
        type=str,
        default='/home/aboggust/code/chart_captioning/vistext/data/data',
        help='Data directory path containing the data files and images and features directories.'
    )
    parser.add_argument(
        '--split', 
        type=str, 
        default='test', 
        choices=['train', 'val', 'test'],
        help='Data split to extract features for. Expects data file to be named data_{split}.json'
    )
    args = parser.parse_args()
    
    data_dir = Path(args.data_dir).resolve()
    img_dir = data_dir.joinpath('images')
    out_dir = data_dir.joinpath('features')
    
    print('Loading images from:', img_dir)

    data_file = os.path.join(data_dir, f'data_{args.split}.json')
    dataset = VisTextImageDataset(data_file, img_dir)
    dataloader = DataLoader(dataset, batch_size=1,
                            shuffle=False, collate_fn=collate_fn, num_workers=4)
    
    print(f'{len(dataset)} {args.split} images are being processed.')

    output_fname = out_dir.joinpath(f'features_{args.split}_boxes{NUM_OBJECTS}.h5')
    print('Saving features to:', output_fname)

    desc = f'vistext_{args.split}_{(NUM_OBJECTS, DIM)}'
    extract(output_fname, dataloader, desc)
