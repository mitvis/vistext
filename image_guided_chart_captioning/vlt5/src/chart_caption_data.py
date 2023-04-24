# VisText chart captioning Dataset and DataLoader.

import h5py
import json
import language_evaluation
import math
import numpy as np
import os
import pickle
import sacrebleu
import torch
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.distributed import DistributedSampler

from vlt5.src.tokenization import VLT5TokenizerFast


class ChartCaptionFineTuneDataset(Dataset):
    """VisText Dataset for loading charts and captions."""
    
    def __init__(self, args, split):
        """
        Args:
            args (argparse Config): VisText arguments.
            split (string): Loads the given split. Valid options are 'train', 
                'val', or 'test'.
        """
        self.args = args
        
        data_file = os.path.join(args.data_directory, f'data_{split}.json')
        with open(data_file, 'r') as f:
            self.data = json.load(f)
                    
        self.features_file = os.path.join(args.data_directory, 'features', f'features_{split}_boxes36.h5')
        
        self.args.tokenizer = self.args.backbone
        self.tokenizer = VLT5TokenizerFast.from_pretrained(
            self.args.backbone,
            max_length=self.args.max_text_length,
            do_lower_case=self.args.do_lower_case,
        )
        
    def __len__(self):
        """Returns the integer length of the dataset. Each data instance is a 
        chart and caption pair. If prefix tuning, the length of the dataset is 
        doubled because each chart is paired with its L1 and L2L3 captions 
        seperately."""
        
        if self.args.prefix_tuning:
            return len(self.data) * 2
        return len(self.data)
    
    def __getitem__(self, idx):
        """
        Gets a dataset item at index idx.
        
        Args:
            idx (int): index in the Dataset. Must be greater than 0 and less
                than the length of the Dataset.
            
        Returns: A dictionary containing:
            id (string): data item ID
            img_id (string): image ID
            img_path (string): path to the chart image
            n_boxes (int): number of visual feature boxes
            boxes (torch.Tensor): visual features boxes
            vis_feats (torch.Tensor): visual features
            input_text (torch.Tensor): input text from scene graph or data table
            input_ids (torch.Tensor): tokenized input text
            input_length (torch.Tensor): number of input tokens
            target_text (torch.Tensor): caption text
            target_ids (torch.Tensor): tokenized caption text
            target_length (torch.Tensor): number of caption tokens.
        """
        output = {}
        
        # For prefix tuning, we split each data instance into two data 
        # instances: a (chart, L1 caption) and a (chart, L2 caption).
        if self.args.prefix_tuning:
            data_sample = self.data[math.floor(idx/2)]
        else:
            data_sample = self.data[idx]
        
        output['id'] = data_sample['caption_id']
        
        img_id = data_sample['img_id']
        output['img_id'] = img_id
        output['img_path'] = f"images/{data_sample['img_id']}.png"
        
        # Visual Features: normalize them between 0 and 1 and add to the output.
        f = h5py.File(self.features_file, 'r')
        img_h = f[f'{img_id}/img_h'][()]
        img_w = f[f'{img_id}/img_w'][()]
        boxes = f[f'{img_id}/boxes'][()]  # (x1, y1, x2, y2)
        boxes[:, (0, 2)] /= img_w
        boxes[:, (1, 3)] /= img_h
        np.testing.assert_array_less(boxes, 1+1e-5)
        np.testing.assert_array_less(-boxes, 0+1e-5)
        boxes = torch.from_numpy(boxes)
        boxes.clamp_(min=0.0, max=1.0)
        n_boxes = len(boxes)

        feats = np.zeros(shape=(n_boxes, 2048), dtype=np.float32)
        f[f'{img_id}/features'].read_direct(feats)
        feats = torch.from_numpy(feats)

        n_boxes = min(n_boxes, self.args.max_n_boxes)
        output['n_boxes'] = n_boxes
        output['boxes'] = boxes[:n_boxes]
        output['vis_feats'] = feats[:n_boxes]
        
        # Text Input: process textual chart represenation and add prefix.
        prefix = 'translate chart to L1L2L3: '
        if self.args.prefix_tuning:
            # If index is even get the L1 caption otherwise use L2L3.
            if idx % 2 == 0: 
                prefix = 'translate chart to L1: '
            else:
                prefix = 'translate chart to L2L3: '
        
        input_text = prefix
        if self.args.input_type != 'imageonly':
            input_text += data_sample[self.args.input_type]
        input_ids = self.tokenizer.encode(input_text,
                                          max_length=self.args.max_text_length, 
                                          truncation=True)
        output['input_text'] = input_text
        output['input_ids'] = torch.LongTensor(input_ids)
        output['input_length'] = len(input_ids)      
        
        
        # Chart Caption: Load the target caption.
        target_text = f"{data_sample['caption_L1']} {data_sample['caption_L2L3']}"
        if self.args.prefix_tuning:
            # If index is even get the L1 caption otherwise use L2L3.
            if idx % 2 == 0:
                target_text = data_sample['caption_L1']
                output['id'] += '_L1'
            else:
                target_text = data_sample['caption_L2L3']
                output['id'] += '_L2L3'
        target_ids = self.tokenizer.encode(target_text, 
                                           max_length=self.args.gen_max_length, 
                                           truncation=True)
        output['target_text'] = target_text
        output['target_ids'] = torch.LongTensor(target_ids)
        output['target_length'] = len(target_ids)
                        
        return output
    

    def collate_fn(self, batch):
        """Custom collation for batching (chart, caption) pairs."""
        batch_entry = {}

        B = len(batch)

        S_W_L = max(entry['input_length'] for entry in batch)
        input_ids = torch.ones(B, S_W_L, dtype=torch.long) * self.tokenizer.pad_token_id

        if self.args.use_vision:
            V_L = max(entry['n_boxes'] for entry in batch)
            feat_dim = batch[0]['vis_feats'].shape[-1]

            boxes = torch.zeros(B, V_L, 4, dtype=torch.float)
            vis_feats = torch.zeros(B, V_L, feat_dim, dtype=torch.float)
            vis_attention_mask = torch.zeros(B, V_L, dtype=torch.float)

        if 'target_ids' in batch[0]:
            T_W_L = max(entry['target_length'] for entry in batch)
            target_ids = torch.ones(B, T_W_L, dtype=torch.long) * self.tokenizer.pad_token_id

        targets = []
        img_ids = []
        input_text = []
        ids = []

        for i, entry in enumerate(batch):
            ids.append(entry['id'])
            input_ids[i, :entry['input_length']] = entry['input_ids']

            if self.args.use_vision:
                n_boxes = entry['n_boxes']
                boxes[i, :n_boxes] = entry['boxes']
                vis_feats[i, :n_boxes] = entry['vis_feats']
                vis_attention_mask[i, :n_boxes] = 1
                img_ids.append(entry['img_id'])

            target_ids[i, :entry['target_length']] = entry['target_ids']
            
            input_text.append(entry['input_text'])
            targets.append(entry['target_text'])

        batch_entry['ids'] = ids
        batch_entry['input_ids'] = input_ids
        if 'target_ids' in batch[0]:
            word_mask = target_ids != self.tokenizer.pad_token_id
            target_ids[~word_mask] = -100
            batch_entry['target_ids'] = target_ids

        if self.args.use_vision:
            batch_entry['boxes'] = boxes
            batch_entry['vis_feats'] = vis_feats
            batch_entry['vis_attention_mask'] = vis_attention_mask
            batch_entry['img_id'] = img_ids

        batch_entry['input_text'] = input_text
        batch_entry['targets'] = targets
        batch_entry['task'] = 'caption'

        return batch_entry
    
    
def get_loader(args, split, mode, batch_size, workers, 
               distributed, gpu):
    """Create a VisText DataLoader.
    
    Args:
        args (argparse Config): VisText arguments.
        split (string): The data split to load. Options are 'train', 'val', and 
            'test'.
        mode (string): The mode to load the data in. Options are 'train' and 
            'val'. 'train' adds shuffling to the DataLoader while 'val' does 
            not. 
        batch_size (int): The size of each batch.
        workers (int): The number of workers.
        distributed (boolean): If the DataLoader is distributed across GPUs.
        gpu (int): The GPU index.
        
    Returns: a PyTorch DataLoader for the VisText dataset.
    """

    verbose = (gpu == 0) # Only be verbose for one GPU.

    dataset = ChartCaptionFineTuneDataset(
        args=args,
        split=split,
    )

    if distributed:
        sampler = DistributedSampler(dataset)
    else:
        sampler = None

    if mode == 'train':
        loader = DataLoader(
            dataset, 
            batch_size=batch_size, 
            num_workers=workers, 
            pin_memory=True, 
            sampler=sampler,
            shuffle=(sampler is None),
            collate_fn=dataset.collate_fn
        )
    else:
        loader = DataLoader(
            dataset,
            batch_size=batch_size,
            num_workers=workers, 
            pin_memory=True,
            sampler=sampler,
            shuffle=None if (sampler is not None) else False,
            collate_fn=dataset.collate_fn,
            drop_last=False)

    loader.task = 'chart_caption'
    return loader
