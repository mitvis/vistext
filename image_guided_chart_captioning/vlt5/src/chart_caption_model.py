# VisText chart captioning model class.

import torch
import torch.nn

from vlt5.src.modeling_bart import VLBart
from vlt5.src.modeling_t5 import VLT5


class VLT5ChartCaption(VLT5):
    """VisText chart captioning model with a VLT5 backbone."""

    def __init__(self, config):
        super().__init__(config)

    def train_step(self, batch):
        """Model training step. Returns the training loss."""
        device = next(self.parameters()).device
        vis_feats = batch['vis_feats'].to(device)
        input_ids = batch['input_ids'].to(device)
        vis_pos = batch['boxes'].to(device)
        lm_labels = batch['target_ids'].to(device)

        output = self(
            input_ids=input_ids,
            vis_inputs=(vis_feats, vis_pos),
            labels=lm_labels,
            reduce_loss=True,
        )

        lm_mask = lm_labels != -100
        B, L = lm_labels.size()

        result = {
            'loss': output['loss']
        }
        return result

    def test_step(self, batch, predict=True, **kwargs):
        """Model testing step. Returns the test loss and predictions."""
        device = next(self.parameters()).device
        vis_feats = batch['vis_feats'].to(device)
        input_ids = batch['input_ids'].to(device)
        vis_pos = batch['boxes'].to(device)
        lm_labels = batch['target_ids'].to(device)

        output = self(
            input_ids=input_ids,
            vis_inputs=(vis_feats, vis_pos),
            labels=lm_labels,
            reduce_loss=True,
        )

        result = {
            'loss': output['loss'],
        }

        if predict:
            prediction = self.generate(
                input_ids=input_ids,
                vis_inputs=(vis_feats, vis_pos),
                **kwargs,
            )
            generated_caption = self.tokenizer.batch_decode(prediction, 
                                                            skip_special_tokens=True)
            result['pred'] =  generated_caption

        return result


class VLBartChartCaption(VLBart):
    """VisText chart captioning model with a Bart backbone."""

    def __init__(self, config):
        super().__init__(config)

    def train_step(self, batch):
        """Model training step. Returns the training loss."""
        device = next(self.parameters()).device
        vis_feats = batch['vis_feats'].to(device)
        input_ids = batch['input_ids'].to(device)
        vis_pos = batch['boxes'].to(device)
        lm_labels = batch['target_ids'].to(device)

        output = self(
            input_ids=input_ids,
            vis_inputs=(vis_feats, vis_pos),
            labels=lm_labels,
            reduce_loss=True
        )

        lm_mask = lm_labels != -100
        B, L = lm_labels.size()

        result = {
            'loss': output['loss']
        }
        return result

    def test_step(self, batch, predict=True, **kwargs):
        """Model testing step. Returns the test loss and predictions."""
        device = next(self.parameters()).device
        vis_feats = batch['vis_feats'].to(device)
        input_ids = batch['input_ids'].to(device)
        vis_pos = batch['boxes'].to(device)
        lm_labels = batch['target_ids'].to(device)

        output = self.generate(
            input_ids=input_ids,
            vis_inputs=(vis_feats, vis_pos),
            **kwargs
        )

        output = self(
            input_ids=input_ids,
            vis_inputs=(vis_feats, vis_pos),
            labels=lm_labels,
            reduce_loss=True,
        )

        result = {
            'loss': output['loss'],
        }

        if predict:
            prediction = self.generate(
                input_ids=input_ids,
                vis_inputs=(vis_feats, vis_pos),
                **kwargs,
            )
            generated_caption = self.tokenizer.batch_decode(prediction,
                                                            skip_special_tokens=True)
            result['pred'] =  generated_caption

        return result
