# Parameters for training and evaluating chart captioning models.
# Adapted from https://github.com/j-min/VL-T5.

import argparse
import numpy as np
import pprint
import random
import torch
import yaml


def str2bool(string):
    """Convert string to python Boolean value."""
    if string.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif string.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


def get_optimizer(optim, verbose=False):
    """Create optimizer from optim string.
    Args:
        optim (string): Name of the optimizer to use. Valid options are: 'rms', 
            'adam', 'adamw', 'adamax', or 'sgd'. ValueError is thrown if invalid
            option is passed.
        verbose (boolean): If True, the name of the optimizer is printed out.
    
    Returns:
        Created optimizer function. If optimixer is adamw, the string 'adamw' is
        returned and turned into an optimizer function in trainer_base.py.
    """
    if optim == 'rms':
        if verbose:
            print("Optimizer: Using RMSProp")
        optimizer = torch.optim.RMSprop
    elif optim == 'adam':
        if verbose:
            print("Optimizer: Using Adam")
        optimizer = torch.optim.Adam
    elif optim == 'adamw':
        if verbose:
            print("Optimizer: Using AdamW")
        optimizer = 'adamw'
    elif optim == 'adamax':
        if verbose:
            print("Optimizer: Using Adamax")
        optimizer = torch.optim.Adamax
    elif optim == 'sgd':
        if verbose:
            print("Optimizer: SGD")
        optimizer = torch.optim.SGD
    else:
        raise ValueError(f"Please add your optimizer {optim} to the function.")
    return optimizer


def parse_args(parse=True, **optional_kwargs):
    """Parse command line arguments. Returns argparse Config."""
    parser = argparse.ArgumentParser()

    parser.add_argument('--seed', type=int, default=9595, help='Random seed.')

    # Data
    parser.add_argument('--data_directory', type=str, default='data/data/',
                        help='Path to data files.')
    parser.add_argument('--train', action='store_true',
                        help='Train the model.')
    parser.add_argument('--predict', action='store_true',
                        help='Output val and test predictions.')

    # Checkpoint
    parser.add_argument('--output', type=str, default='models/', 
                        help='Custom output based on experiment is recommended.'
                       )
    parser.add_argument('--load', type=str, default=None, 
                        help='Load the model.')
    parser.add_argument('--from_scratch', action='store_true', 
                        help='Do not load pretrained weights.')

    # CPU/GPU
    parser.add_argument("--multiGPU", action='store_const', default=False, 
                        const=True, help='Use multiple GPUs.')
    parser.add_argument('--fp16', action='store_true', 
                        help='Use half point floats.')
    parser.add_argument("--distributed", action='store_true', 
                        help='Distribute across GPUs.')
    parser.add_argument("--num_workers", default=4, type=int, 
                        help='Number of workers.')
    parser.add_argument('--local_rank', type=int, default=-1, 
                        help='Local rank.')

    # Model Config
    parser.add_argument('--backbone', type=str, default='t5-base',
                        choices=['t5-base', 'facebook/bart-base'],
                        help='Model backbone to use.')
    parser.add_argument('--tokenizer', type=str, default=None, 
                        help="""Tokenizer to use. If None, it is set based on 
                        the model backbone.""")

    parser.add_argument('--feat_dim', type=float, default=2048, 
                       help='Dimensions of visual features.')
    parser.add_argument('--pos_dim', type=float, default=4, 
                        help='Position embedding dimensions.')

    parser.add_argument('--use_vision', default=True, type=str2bool,
                       help='Use chart images.')
    parser.add_argument('--use_vis_order_embedding', default=True, 
                        type=str2bool)
    parser.add_argument('--use_vis_layer_norm', default=True, type=str2bool)
    parser.add_argument('--individual_vis_layer_norm', default=True, 
                        type=str2bool)
    parser.add_argument('--share_vis_lang_layer_norm', action='store_true')

    parser.add_argument('--n_boxes', type=int, default=36, 
                        help='Number of boxes used to compute visual features.')
    parser.add_argument('--max_n_boxes', type=int, default=36, 
                        help='Maximum boxes used for visual features.')
    parser.add_argument('--max_text_length', type=int, default=1024, 
                        help='Maximum number of textual input tokens allowed.')

    # Training
    parser.add_argument('--batch_size', type=int, default=256)
    parser.add_argument('--optim', default='adamw')
    parser.add_argument('--warmup_ratio', type=float, default=0.1)
    parser.add_argument('--weight_decay', type=float, default=0.01)
    parser.add_argument('--clip_grad_norm', type=float, default=5)
    parser.add_argument('--gradient_accumulation_steps', type=int, default=1)
    parser.add_argument('--lr', type=float, default=5e-5)
    parser.add_argument('--adam_eps', type=float, default=1e-6)
    parser.add_argument('--adam_beta1', type=float, default=0.9)
    parser.add_argument('--adam_beta2', type=float, default=0.999)
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--dropout', type=float, default=0.1)
    parser.add_argument("--losses", default='lm,obj,attr,feat', type=str)

    # Inference
    parser.add_argument('--num_beams', type=int, default=5, 
                        help='Number of beams for beam search.')
    parser.add_argument('--gen_max_length', type=int, default=512, 
                        help='Maximum number of tokens to generate.')

    # Data
    parser.add_argument('--do_lower_case', action='store_true', 
                        help='Convert text to lower case.')
    
    # Chart Captioning
    parser.add_argument('--prefix_tuning', default=True, type=str2bool, 
                        help='Perform semantic prefix tuning.')
    parser.add_argument('--input_type', type=str, default='scenegraph', 
                        choices=['scenegraph', 'datatable', 'imageonly'],
                        help='Textual representation of the chart to use.')
    
    # Parse the arguments.
    if parse:
        args = parser.parse_args()
    else: # For interactive engironmnet (ex. jupyter)
        args = parser.parse_known_args()[0]

    # Convert namespace to dictionary.
    kwargs = vars(args)
    kwargs.update(optional_kwargs)

    args = Config(**kwargs)

    # Bind optimizer class.
    verbose = False
    args.optimizer = get_optimizer(args.optim, verbose=verbose)

    # Set seeds.
    torch.manual_seed(args.seed)
    random.seed(args.seed)
    np.random.seed(args.seed)

    return args


class Config(object):
    def __init__(self, **kwargs):
        """Configuration Class: set kwargs as class attributes with setattr."""
        for k, v in kwargs.items():
            setattr(self, k, v)

    @property
    def config_str(self):
        return pprint.pformat(self.__dict__)

    def __repr__(self):
        """Pretty-print configurations in alphabetical order."""
        config_str = 'Configurations\n'
        config_str += self.config_str
        return config_str

    def save(self, path):
        with open(path, 'w') as f:
            yaml.dump(self.__dict__, f, default_flow_style=False)

    @classmethod
    def load(cls, path):
        with open(path, 'r') as f:
            kwargs = yaml.load(f)

        return Config(**kwargs)


if __name__ == '__main__':
    args = parse_args(True)
