# Functions for tokenization.
# Originally from https://github.com/j-min/VL-T5.

import re
import sentencepiece as spm
from tokenizers import processors
from transformers import T5Tokenizer, T5TokenizerFast, PreTrainedTokenizer, PreTrainedTokenizerBase
from transformers.convert_slow_tokenizer import SpmConverter
from typing import List


class VLT5Tokenizer(T5Tokenizer):
    """A tokenizer for VLT5.
    
    The special tokens of T5Tokenizer is hard-coded as <extra_id_{}>. 
    VLT5Tokenizer extends the special tokens to be <vis_extra_id_{}>.
    """
    def __init__(
        self,
        vocab_file,
        eos_token="</s>",
        unk_token="<unk>",
        pad_token="<pad>",
        extra_ids=100,
        vis_extra_ids=100,
        additional_special_tokens=None,
        **kwargs
    ):
        # Add extra_ids to the special token list.
        if extra_ids > 0 and additional_special_tokens is None:
            additional_special_tokens = ["<extra_id_{}>".format(i) for i in range(extra_ids)]
        elif extra_ids > 0 and additional_special_tokens is not None:
            # Check that we have the right number of extra_id special tokens.
            extra_tokens = len(set(filter(lambda x: bool("extra_id" in x), additional_special_tokens)))
            if extra_tokens != extra_ids:
                raise ValueError(
                    f"""Both extra_ids ({extra_ids}) and 
                    additional_special_tokens ({additional_special_tokens}) are 
                    provided to T5Tokenizer. Additional_special_tokens must 
                    include extra_ids tokens."""
                )

        if vis_extra_ids > 0:
            additional_special_tokens.extend(["<vis_extra_id_{}>".format(i) for i in range(vis_extra_ids)])

        PreTrainedTokenizer.__init__(
            self,
            eos_token=eos_token,
            unk_token=unk_token,
            pad_token=pad_token,
            extra_ids=extra_ids,
            additional_special_tokens=additional_special_tokens,
            **kwargs,
        )

        self.vocab_file = vocab_file
        self._extra_ids = extra_ids
        self._vis_extra_ids = vis_extra_ids

        self.sp_model = spm.SentencePieceProcessor()
        self.sp_model.Load(vocab_file)

    @property
    def vocab_size(self):
        return self.sp_model.get_piece_size() + self._extra_ids + self._vis_extra_ids

    def get_vocab(self):
        vocab = {self.convert_ids_to_tokens(
            i): i for i in range(self.vocab_size)}
        vocab.update(self.added_tokens_encoder)
        return vocab

    def _convert_token_to_id(self, token):
        """ Converts a token (str) in an id using the vocab. """
        if token.startswith("<extra_id_"):
            match = re.match(r"<extra_id_(\d+)>", token)
            num = int(match.group(1))
            return self.vocab_size - num - 1 - self._vis_extra_ids
        elif token.startswith("<vis_extra_id_"):
            match = re.match(r"<vis_extra_id_(\d+)>", token)
            num = int(match.group(1))
            return self.vocab_size - num - 1
        return self.sp_model.piece_to_id(token)

    def _convert_id_to_token(self, index):
        """Converts an index (integer) in a token (str) using the vocab."""
        if index < self.sp_model.get_piece_size():
            token = self.sp_model.IdToPiece(index)
        else:
            if index > self.sp_model.get_piece_size() + self._extra_ids - 1:
                token = "<vis_extra_id_{}>".format(self.vocab_size - 1 - index)
            else:
                token = "<extra_id_{}>".format(self.vocab_size - self._vis_extra_ids - 1 - index)
        return token


class VLT5Converter(SpmConverter):
    def vocab(self, proto):
        vocab = [(piece.piece, piece.score) for piece in proto.pieces]
        num_extra_ids = self.original_tokenizer._extra_ids
        vocab += [("<extra_id_{}>".format(i), 0.0)
                  for i in range(num_extra_ids - 1, -1, -1)]

        num_vis_extra_ids = self.original_tokenizer._vis_extra_ids
        vocab += [("<vis_extra_id_{}>".format(i), 0.0)
                  for i in range(num_vis_extra_ids - 1, -1, -1)]

        return vocab

    def post_processor(self):
        return processors.TemplateProcessing(
            single=["$A", "</s>"],
            pair=["$A", "</s>", "$B", "</s>"],
            special_tokens=[
                ("</s>", self.original_tokenizer.convert_tokens_to_ids("</s>")),
            ],
        )


def convert_slow_vlt5tokenizer(vlt5tokenizer):
    return VLT5Converter(vlt5tokenizer).converted()


class VLT5TokenizerFast(T5TokenizerFast):
    slow_tokenizer_class = VLT5Tokenizer
    prefix_tokens: List[int] = []

    def __init__(
        self,
        vocab_file,
        tokenizer_file=None,
        eos_token="</s>",
        unk_token="<unk>",
        pad_token="<pad>",
        extra_ids=100,
        vis_extra_ids=100,
        additional_special_tokens=None,
        **kwargs
    ):
        # Add extra_ids to the special token list
        if extra_ids > 0 and additional_special_tokens is None:
            additional_special_tokens = ["<extra_id_{}>".format(i) for i in range(extra_ids)]
        elif extra_ids > 0 and additional_special_tokens is not None:
            # Check that we have the right number of extra_id special tokens
            extra_tokens = len(set(filter(lambda x: bool("extra_id" in x), additional_special_tokens)))
            if extra_tokens != extra_ids:
                raise ValueError(
                    f"""Both extra_ids ({extra_ids}) and 
                    additional_special_tokens ({additional_special_tokens}) are 
                    provided to T5Tokenizer. Additional_special_tokens must 
                    include extra_ids tokens."""
                )

        if vis_extra_ids > 0:
            additional_special_tokens.extend(["<vis_extra_id_{}>".format(i) for i in range(vis_extra_ids)])

        slow_tokenizer = self.slow_tokenizer_class(
            vocab_file,
            tokenizer_file=tokenizer_file,
            eos_token=eos_token,
            unk_token=unk_token,
            pad_token=pad_token,
            extra_ids=extra_ids,
            vis_extra_ids=vis_extra_ids,
            # additional_special_tokens=additional_special_tokens,
            **kwargs
        )
        fast_tokenizer = convert_slow_vlt5tokenizer(slow_tokenizer)
        self._tokenizer = fast_tokenizer

        PreTrainedTokenizerBase.__init__(
            self,
            tokenizer_file=tokenizer_file,
            eos_token=eos_token,
            unk_token=unk_token,
            pad_token=pad_token,
            extra_ids=extra_ids,
            vis_extra_ids=vis_extra_ids,
            additional_special_tokens=additional_special_tokens,
            **kwargs,
        )

        self.vocab_file = vocab_file
        self._extra_ids = extra_ids
        self._vis_extra_ids = vis_extra_ids
