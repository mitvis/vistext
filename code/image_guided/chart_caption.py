# VisText chart captioning training and evaluation. 

from datetime import datetime
import json
import logging
import os
from packaging import version
from pathlib import Path
from time import time
import torch
import torch.backends.cudnn as cudnn
import torch.distributed as dist
import torch.nn as nn
from torch.nn.parallel import DistributedDataParallel as DDP
from tqdm import tqdm

from chart_caption_data import get_loader
from chart_caption_model import VLBartChartCaption, VLT5ChartCaption
import dist_utils as dist_utils
from param import parse_args
from utils import LossMeter, set_global_logging_level
from trainer_base import TrainerBase

set_global_logging_level(logging.ERROR, ["transformers"])
proj_dir = Path(__file__).resolve().parent.parent

# Check if Pytorch version >= 1.6 switch between Native AMP and Apex
_use_native_amp = False
_use_apex = False
if version.parse(torch.__version__) < version.parse("1.6"):
    from transformers.file_utils import is_apex_available
    if is_apex_available():
        from apex import amp
        _use_apex = True
else:
    _use_native_amp = True
    from torch.cuda.amp import autocast


class Trainer(TrainerBase):
    """Trainer class for VisText chart captioning."""
    def __init__(self, args, train_loader=None, val_loader=None, 
                 test_loader=None, train=True):
        super().__init__(
            args,
            train_loader=train_loader,
            val_loader=val_loader,
            test_loader=test_loader,
            train=train)

        model_kwargs = {}
        if 't5' in args.backbone:
            model_class = VLT5ChartCaption
        elif 'bart' in args.backbone:
            model_class = VLBartChartCaption

        config = self.create_config()

        self.tokenizer = self.create_tokenizer()
        if 'bart' in self.args.tokenizer:
            num_added_toks = 0
            if config.use_vis_order_embedding:
                additional_special_tokens = [f'<extra_id_{i}>' for i in range(100-1, -1, -1)] + \
                        [f'<vis_extra_id_{i}>' for i in range(100-1, -1, -1)]
                special_tokens_dict = {'additional_special_tokens': additional_special_tokens}
                num_added_toks = self.tokenizer.add_special_tokens(special_tokens_dict)
                config.default_obj_order_ids = self.tokenizer.convert_tokens_to_ids([f'<vis_extra_id_{i}>' for i in range(100)])

        self.model = self.create_model(model_class, config, **model_kwargs)
        if 't5' in self.args.tokenizer:
            self.model.resize_token_embeddings(self.tokenizer.vocab_size)
        elif 'bart' in self.args.tokenizer:
            self.model.resize_token_embeddings(self.model.model.shared.num_embeddings + num_added_toks)

        self.model.tokenizer = self.tokenizer 

        # Load checkpoint or initialize model.
        self.start_epoch = None
        if args.load is not None:
            ckpt_path = args.load
            if ckpt_path[-4:] != '.pth':
                ckpt_path = args.load + '.pth'
            self.load_checkpoint(ckpt_path)
            
        if self.args.from_scratch:
            self.init_weights()

        # Set GPU options.
        print(f'Model Launching at GPU {self.args.gpu}')
        if self.verbose:
            start = time()
        self.model = self.model.to(args.gpu)

        # Create optimizer.
        if train:
            self.optim, self.lr_scheduler = self.create_optimizer_and_scheduler()
            if self.args.fp16 and _use_native_amp:
                self.scaler = torch.cuda.amp.GradScaler()
            elif _use_apex:
                self.model, self.optim = amp.initialize(
                    self.model, self.optim, opt_level='O1', verbosity=self.verbose)
        if args.multiGPU:
            if args.distributed:
                self.model = DDP(self.model, device_ids=[args.gpu],
                                 find_unused_parameters=True
                                 )
        if self.verbose:
            print(f'Initializing the trainer took {time() - start:.1f}s')

        # Save arguments to disk.
        argument_file = os.path.join(self.args.output, 'args.json')
        argument_dict = vars(args)
        with open(argument_file, 'w') as f:
            json.dump(argument_dict, f) 


    def train(self):
        """Train the VisText model."""
        best_val_loss = None
        best_epoch = 0
        if self.verbose:
            loss_meter = LossMeter()

        if self.args.distributed:
            dist.barrier()

        global_step = 0
        epochs = self.args.epochs

        for epoch in range(epochs):
            self.model.train()
            if self.start_epoch is not None:
                epoch += self.start_epoch
            if self.args.distributed:
                self.train_loader.sampler.set_epoch(epoch)
            if self.verbose:
                pbar = tqdm(total=len(self.train_loader), ncols=120)

            for step_i, batch in enumerate(self.train_loader):
                # Forward pass.
                if self.args.fp16 and _use_native_amp:
                    with autocast():
                        if self.args.distributed:
                            results = self.model.module.train_step(batch)
                        else:
                            results = self.model.train_step(batch)
                else:
                    if self.args.distributed:
                        results = self.model.module.train_step(batch)
                    else:
                        results = self.model.train_step(batch)
                loss = results['loss']

                # Backwards pass.
                if self.args.fp16 and _use_native_amp:
                    self.scaler.scale(loss).backward()
                elif self.args.fp16 and _use_apex:
                    with amp.scale_loss(loss, self.optim) as scaled_loss:
                        scaled_loss.backward()
                else:
                    loss.backward()
                loss = loss.detach()

                # Update model parameters.
                if self.args.clip_grad_norm > 0:
                    if self.args.fp16 and _use_native_amp:
                        self.scaler.unscale_(self.optim)
                        torch.nn.utils.clip_grad_norm_(
                            self.model.parameters(), self.args.clip_grad_norm)
                    elif self.args.fp16 and _use_apex:
                        torch.nn.utils.clip_grad_norm_(amp.master_params(
                            self.optim), self.args.clip_grad_norm)
                    else:
                        torch.nn.utils.clip_grad_norm_(
                            self.model.parameters(), self.args.clip_grad_norm)

                update = True
                if self.args.gradient_accumulation_steps > 1:
                    if step_i == 0:
                        update = False
                    elif step_i % self.args.gradient_accumulation_steps == 0 or step_i == len(self.train_loader) - 1:
                        update = True
                    else:
                        update = False

                if update:
                    if self.args.fp16 and _use_native_amp:
                        self.scaler.step(self.optim)
                        self.scaler.update()
                    else:
                        self.optim.step()

                    if self.lr_scheduler:
                        self.lr_scheduler.step()
                    for param in self.model.parameters():
                        param.grad = None
                    global_step += 1

                if self.lr_scheduler:
                    if version.parse(torch.__version__) >= version.parse("1.4"):
                        lr = self.lr_scheduler.get_last_lr()[0]
                    else:
                        lr = self.lr_scheduler.get_lr()[0]
                else:
                    try:
                        lr = self.optim.get_lr()[0]
                    except AttributeError:
                        lr = self.args.lr

                if self.verbose:
                    loss_meter.update(loss.item())
                    desc_str = f'Epoch {epoch} | LR {lr:.6f} | Steps {global_step}'
                    desc_str += f' | Loss {loss_meter.val:4f}'
                    pbar.set_description(desc_str)
                    pbar.update(1)

            if self.args.distributed:
                dist.barrier()

            if self.verbose:
                pbar.close()

            # Compute validation loss.
            val_results = self.validate(self.val_loader)
            val_loss = val_results['loss'].detach().cpu() # Detach to prevent memory errors.

            if self.args.distributed: # Average across GPUs.
                dist.barrier()
                dist_val_losses = dist_utils.all_gather(val_loss)
                val_loss = 0
                for dist_val_loss in dist_val_losses:
                    val_loss += dist_val_loss
                val_loss /= len(dist_val_losses)
                # print(f'Setting Barrier gpu {args.gpu}')
                dist.barrier()

            if best_val_loss is None or val_loss < best_val_loss:
                best_val_loss = val_loss
                best_epoch = epoch

            if self.args.distributed:
                dist.barrier()

            if self.verbose:
                print(f'\nEpoch {epoch} Val Loss {val_loss:0.4f}')
                print(f'Best Epoch {best_epoch} Best Val Loss {best_val_loss:0.4f}\n')

            # Save model for this epoch.
            if self.verbose:
                self.save(f"epoch_{epoch:02}")

            if self.args.distributed:
                dist.barrier()


        # Rename the best model.
        if self.verbose:
            best_epoch_path = os.path.join(self.args.output, f"epoch_{best_epoch:02}")
            self.load(best_epoch_path)
            self.save("BEST")

        if self.args.distributed:
            dist.barrier()

    def validate(self, loader):
        """Validate the VisText model."""
        self.model.eval()
        val_results = {'loss': 0}

        if self.args.distributed:
            dist.barrier()

        if self.verbose:
            val_loss_meter = LossMeter()
            val_pbar = tqdm(total=len(loader), ncols=120)

        with torch.no_grad():
            for step_i, batch in enumerate(loader):
                if self.args.fp16 and _use_native_amp:
                    with autocast():
                        if self.args.distributed:
                            batch_results = self.model.module.test_step(batch, predict=False)
                        else:
                            batch_results = self.model.test_step(batch, predict=False)
                else:
                    if self.args.distributed:
                        batch_results = self.model.module.test_step(batch, predict=False)
                    else:
                        batch_results = self.model.test_step(batch, predict=False)  
                        
                val_results['loss'] += batch_results['loss']

                if self.verbose:
                    val_loss_meter.update(batch_results['loss'])
                    desc_str = f'Validation | Loss {val_loss_meter.val:4f}'
                    val_pbar.set_description(desc_str)
                    val_pbar.update(1)

        if self.args.distributed:
            dist.barrier()

        if self.verbose:
            val_pbar.close()  

        val_results['loss'] /= (step_i + 1)
        return val_results

    def predict(self, loader):
        """ Predict chart captions for every chart in the loader.

        Args:
        loader (DataLoader): the DataLoader for a ChartCaptionFineTune Dataset
          containing VisText charts.

        Returns: a dictionary maping the data ids to the predicted captions.
        """
        self.model.eval()
        with torch.no_grad():
            gen_kwargs = {}
            gen_kwargs['num_beams'] = self.args.num_beams
            gen_kwargs['max_length'] = self.args.gen_max_length

            results = {} # Maps data id to predicted chart caption.

            # Get caption predictions.
            desc=f'{loader.dataset.split.title()} Predictions'
            for i, batch in enumerate(tqdm(loader, ncols=120, desc=desc, disable=not self.verbose)):
                if self.args.distributed:
                    output = self.model.module.test_step(
                        batch,
                        predict=True,
                        **gen_kwargs)
                else:
                    output = self.model.test_step(
                        batch,
                        predict=True,
                        **gen_kwargs)

                batch_predictions = output['pred']
                batch_ids = batch['ids']
                batch_results = dict(zip(batch_ids, batch_predictions))

                results.update(batch_results)

            # Handle distributed GPUs.
            if self.args.distributed:
                dist.barrier()
                dist_results = dist_utils.all_gather(results)
                results = {}
                for dist_result in dist_results:
                    results.update(dist_result)
                dist.barrier()
                
            return results

    def predict_and_save(self, output_filename, loader):
        """ Compute predictions for every item in loader and write the
        predictions in dataset order to a newline-delimited .txt file named
        output_filename.

        Args:
        output_filename (str): The name of the output file the predictions are
          written to.
        loader (DataLoader): DataLoader for the data to be predicted.

        Returns: None. Writes a newline-delimited .txt file named
        output_filename to disk containing the predictions in dataset order.
        """
        prediction_results = self.predict(loader)
        dataset = loader.dataset
        if self.verbose: # Only write out to disk from one computation thread.
            predictions = [prediction_results[datum['id']] for datum in dataset]
            with open(output_filename, 'w') as f:
                for i, prediction in enumerate(predictions):
                    line = f'{prediction}\n'
                    if i == len(predictions) - 1:
                        line = f'{prediction}'
                    f.write(line)


def main_worker(gpu, args):
    """Train the VisText model and generate predictions for the validation and
    test data.

    Args:
        gpu (int): The GPU number computation is done on.
        args (argparse Config): The training and evaluation parameters.

    Returns: None. Write the arguments to disk. Trains the model, saving every
    epoch's state dict and the state dict of the model with the lowest
    validation loss to disk. Generates captions for the validation and test data
    and writes them out as a txt file in dataset order. Everything is written to
    the args.output directory.
    """
    args.gpu = gpu
    args.rank = gpu
    print(f'Process Launching at GPU {gpu}')

    if args.distributed:
        torch.cuda.set_device(args.gpu)
        dist.init_process_group(backend='nccl')

    print(f'Building train loader at GPU {gpu}')
    train_loader = get_loader(
        args,
        split='train', mode='train', batch_size=args.batch_size,
        distributed=args.distributed, gpu=args.gpu,
        workers=args.num_workers,
    )

    print(f'Building val loader at GPU {gpu}')
    val_loader = get_loader(
        args,
        split='validation', mode='val', batch_size=args.batch_size,
        distributed=args.distributed, gpu=args.gpu,
        workers=4,
    )

    print(f'Building test loader at GPU {gpu}')
    test_loader = get_loader(
        args,
        split='test', mode='val', batch_size=args.batch_size,
        distributed=args.distributed, gpu=args.gpu,
        workers=4,
        )

    if args.train:
        trainer = Trainer(args, train_loader, val_loader, test_loader, train=True) 
        trainer.train()

    if args.predict:
        model_file = os.path.join(args.output, 'BEST.pth')
        args.load = model_file
        trainer = Trainer(args, train_loader, val_loader, test_loader, train=False)

        val_predictions_filename = os.path.join(args.output, 'val_predictions.txt')
        trainer.predict_and_save(val_predictions_filename, val_loader)

        test_predictions_filename = os.path.join(args.output, 'test_predictions.txt')
        trainer.predict_and_save(test_predictions_filename, test_loader)


if __name__ == "__main__":
    cudnn.benchmark = True
    args = parse_args()
    ngpus_per_node = torch.cuda.device_count()
    args.world_size = ngpus_per_node
    if args.local_rank in [0, -1]:
        print(args)

        comment = None
        if args.load is not None:
            ckpt_str = "_".join(args.load.split('/')[-3:])
            comment = ckpt_str

        current_time = datetime.now().strftime('%b%d_%H-%M')

        run_name = f'{current_time}_GPU{args.world_size}'
        if comment is not None:
            run_name += f'_{comment}'

        args.run_name = run_name

    if args.distributed:
        main_worker(args.local_rank, args)
