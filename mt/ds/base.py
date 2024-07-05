import os
from datasets import DatasetDict, Dataset
import torch
from torch.utils.data import DataLoader
from transformers import (
    PreTrainedTokenizerFast,
)
from tokenizers import (
    Tokenizer,
    decoders,
    models,
    normalizers,
    pre_tokenizers,
    trainers,
)
from tokenizers.processors import TemplateProcessing
from transformers import PreTrainedTokenizerFast


class BaseDataset:
    name: str
    dataset: DatasetDict
    source_lang: str
    target_lang: str
    is_encoder_decoder: bool = False

    def __init__(self, source, target, is_encoder_decoder):
        pass

    def get_ckpt_path(self, model_name):
        pass

    def get_tokenizer(self) -> PreTrainedTokenizerFast:
        source, target = (
            (self.source_lang, self.target_lang)
            if self.source_lang == "en"
            else (self.target_lang, self.source_lang)
        )

        dir = os.path.join("data/mt/tokenizers", f"{self.name}-{source}-{target}")
        special_tokens = {
            "pad_token": "<PAD>",
            "eos_token": "<EOS>",
            "sep_token": "<SEP>",
            "bos_token": "<BOS>",
        }

        assert os.path.exists(
            dir
        ), f"Tokenizer for {self.source_lang}-{self.target_lang} not found."

        tokenizer = PreTrainedTokenizerFast.from_pretrained(dir)
        tokenizer.add_special_tokens(special_tokens)
        tokenizer.padding_side = "left"
        tokenizer.model_max_length = 4096  # FIXME needed?
        return tokenizer

    def build_tokenizer(self, vocab_size=32000):
        tokenizer = Tokenizer(models.BPE())
        tokenizer.normalizer = normalizers.NFKC()
        tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel()
        tokenizer.decoder = decoders.ByteLevel()
        trainer = trainers.BpeTrainer(
            vocab_size=vocab_size,
            initial_alphabet=pre_tokenizers.ByteLevel.alphabet(),
            special_tokens=["<PAD>", "<EOS>", "<SEP>", "<BOS>"],
        )

        it = self.get_data_iterator()
        tokenizer.train_from_iterator(it, trainer=trainer)

        tokenizer.enable_padding(pad_id=0, pad_token="<PAD>", direction="left")
        tokenizer.post_processor = TemplateProcessing(
            single="$A <EOS>",
            pair="$A <SEP> $B:1 <EOS>",
            special_tokens=[
                ("<SEP>", 2),
                ("<EOS>", 1),
            ],
        )
        tok = PreTrainedTokenizerFast(tokenizer_object=tokenizer)

        source, target = (
            (self.source_lang, self.target_lang)
            if self.source_lang == "en"
            else (self.target_lang, self.source_lang)
        )

        path = os.path.join("data/mt/tokenizers", f"{self.name}-{source}-{target}")
        tok.save_pretrained(path)

    def get_data_iterator(self):
        """returns an iterator over the dataset"""
        pass

    def get_dataloader(
        self,
        ds: Dataset,
        fn: callable,
        fn_kwargs: dict,
        batch_size: int,
        columns: list,
        remove_columns: list,
        collate_fn: callable,
        num_workers=2,
        shuffle=True,
    ):
        processed_ds = ds.map(
            fn,
            remove_columns=remove_columns,
            fn_kwargs=fn_kwargs,
            batched=True,
        )
        processed_ds.set_format(type="torch", columns=columns)
        return DataLoader(
            processed_ds,
            batch_size=batch_size,
            collate_fn=collate_fn,
            shuffle=shuffle,
            generator=torch.Generator().manual_seed(42) if shuffle else None,
            num_workers=num_workers,
        )

    def get_dataloaders(
        self,
        tokenizer: PreTrainedTokenizerFast,
        train_fn: callable = None,
        train_fn_kwargs: dict = None,
        val_fn: callable = None,
        val_fn_kwargs: dict = None,
        test_fn: callable = None,
        test_fn_kwargs: dict = None,
        train_batch_size: int = 32,
        val_batch_size: int = 64,
        test_batch_size: int = 64,
        test=False,
    ):
        pass
