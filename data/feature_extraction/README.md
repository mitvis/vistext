# Feature Extraction

This repo contains code to extract visual features compatible with [VL-T5](https://github.com/j-min/VL-T5).

We extract features uing [Hao Tan's Detectron2 implementation of 'Bottom-Up Feature Extractor'](https://github.com/airsplay/py-bottom-up-attention), which is compatible with [the original Caffe implementation](https://github.com/peteanderson80/bottom-up-attention).

Following LXMERT, we use the feature extractor which outputs 36 boxes per image.
We store features in hdf5 format.

## Download VisText Visual Features

To download the visual features we extracted from the VisText dataset: TODO.

## Optional: Manually Extract Visual Features

To extract the VisText visual features yourself:

### 1. Install the feature extractor. 
Please see [the original installation guide](https://github.com/airsplay/py-bottom-up-attention#installation).

### 2. Manually extract and convert the features. 
`vistext_proposal.py` will extract features from 36 detected bounding boxes. 

Run `python vistext_proposal.py --data_dir {data_dir} --split {split}`. 
* `data_dir` should be the path to your `data` folder containing an `images` subfolder with the VisText chart images and a `features` subfolder where the visual feature hdf5 files will be written. 
* `split` is one of `"train"`, `"val"`, or `"test"`. Visual features will be extracted for that split.
