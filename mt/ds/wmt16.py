import os
from datasets import load_dataset, DatasetDict
from transformers import (
    PreTrainedTokenizerFast,
    DataCollatorForSeq2Seq,
    DataCollatorWithPadding,
)
from transformers import PreTrainedTokenizerFast

from utils.mt import DataCollatorDecForSeq2Seq
from mt.ds.base import BaseDataset


class WMT16Dataset(BaseDataset):
    name: str = "wmt16"
    dataset: DatasetDict = None
    source_lang: str = None
    target_lang: str = None
    is_encoder_decoder: bool = False

    def __init__(self, source, target, is_encoder_decoder):
        self.is_encoder_decoder = is_encoder_decoder
        self.source_lang, self.target_lang = source, target

        # https://huggingface.co/datasets/stas/wmt16-en-ro-pre-processed
        # stas/wmt16-en-ro-pre-processed

        # self.dataset = load_dataset(
        #    "stas/wmt16-en-ro-pre-processed",
        # )

        source, target = (
            (self.source_lang, self.target_lang)
            if self.source_lang != "en"
            else (self.target_lang, self.source_lang)
        )

        self.dataset = load_dataset(
            "wmt16",
            f"{source}-{target}",
        )

    def get_ckpt_path(self, model_name):
        return os.path.join(
            f"data/mt/{self.name}-{model_name}",
            f"{self.source_lang}-{self.target_lang}",
        )

    def get_data_iterator(self):
        for i in range(0, len(self.dataset["train"])):
            yield self.dataset["train"][i]["translation"][
                self.source_lang
            ] + " " + self.dataset["train"][i]["translation"][self.target_lang]

        for i in range(0, len(self.dataset["validation"])):
            yield self.dataset["validation"][i]["translation"][
                self.source_lang
            ] + " " + self.dataset["validation"][i]["translation"][self.target_lang]

        for i in range(0, len(self.dataset["test"])):
            yield self.dataset["test"][i]["translation"][
                self.source_lang
            ] + " " + self.dataset["test"][i]["translation"][self.target_lang]

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
        train_dl, val_dl, test_dl = None, None, None

        kwargs = {
            "tokenizer": tokenizer,
        }

        if self.is_encoder_decoder:
            train_fn = self.encdec_training_preprocess
            val_fn = self.encdec_validation_preprocess
            train_fn_kwargs = val_fn_kwargs = kwargs
            train_columns = val_columns = ["input_ids", "labels"]
            tokenizer.padding_side = "right"
            t_collate_fn = v_collate_fn = DataCollatorForSeq2Seq(
                tokenizer=tokenizer,
                label_pad_token_id=tokenizer.pad_token_id,
            )

        else:
            train_fn = self.training_preprocess
            val_fn = self.validation_preprocess
            train_fn_kwargs = val_fn_kwargs = kwargs
            train_columns = ["input_ids"]
            val_columns = ["input_ids", "labels"]
            t_collate_fn = DataCollatorWithPadding(tokenizer=tokenizer)
            # in this scenario pad ids left and labels right
            v_collate_fn = DataCollatorDecForSeq2Seq(
                tokenizer=tokenizer,
                label_pad_token_id=tokenizer.pad_token_id,
            )

        if not test:
            train_dl = self.get_dataloader(
                self.dataset["train"],
                fn=train_fn,
                fn_kwargs=train_fn_kwargs,
                batch_size=train_batch_size,
                columns=train_columns,
                remove_columns=["translation"],
                collate_fn=t_collate_fn,
            )

            val_dl = self.get_dataloader(
                self.dataset["validation"],
                fn=val_fn,
                fn_kwargs=val_fn_kwargs,
                batch_size=val_batch_size,
                columns=val_columns,
                remove_columns=["translation"],
                collate_fn=v_collate_fn,
                shuffle=False,
            )

        else:
            test_dl = self.get_dataloader(
                self.dataset["test"],
                fn=val_fn,
                fn_kwargs=val_fn_kwargs,
                batch_size=test_batch_size,
                columns=val_columns,
                remove_columns=["translation"],
                collate_fn=v_collate_fn,
                shuffle=False,
            )

        return train_dl, val_dl, test_dl

    def training_preprocess(
        self,
        batch,
        tokenizer: PreTrainedTokenizerFast,
    ):
        source_sentences = [
            f"{sample[self.source_lang]}{tokenizer.sep_token}{sample[self.target_lang]}"
            for sample in batch["translation"]
        ]

        source_tokenized = tokenizer(source_sentences, truncation=True, max_length=512)

        return {
            "input_ids": source_tokenized["input_ids"],
        }

    def encdec_training_preprocess(
        self,
        batch,
        tokenizer: PreTrainedTokenizerFast,
    ):
        source_sentences = [sample[self.source_lang] for sample in batch["translation"]]
        target_sentences = [
            tokenizer.bos_token + sample[self.target_lang]
            for sample in batch["translation"]
        ]

        source_tokenized = tokenizer(source_sentences)
        target_tokenized = tokenizer(target_sentences)

        return {
            "input_ids": source_tokenized["input_ids"],
            "labels": target_tokenized["input_ids"],
        }

    def validation_preprocess(
        self,
        batch,
        tokenizer: PreTrainedTokenizerFast,
    ):
        source_sentences = [
            f"{sample[self.source_lang]}{tokenizer.sep_token}"
            for sample in batch["translation"]
        ]

        target_sentences = [sample[self.target_lang] for sample in batch["translation"]]

        source_tokenized = tokenizer(source_sentences, truncation=True, max_length=256)
        # remove eos
        source_tokenized["input_ids"] = [
            sample[:-1] for sample in source_tokenized["input_ids"]
        ]
        target_tokenized = tokenizer(target_sentences, truncation=True, max_length=256)
        return {
            "input_ids": source_tokenized["input_ids"],
            "labels": target_tokenized["input_ids"],
        }

    def encdec_validation_preprocess(
        self,
        batch,
        tokenizer: PreTrainedTokenizerFast,
    ):
        source_sentences = [sample[self.source_lang] for sample in batch["translation"]]
        target_sentences = [
            tokenizer.bos_token + sample[self.target_lang]
            for sample in batch["translation"]
        ]

        source_tokenized = tokenizer(source_sentences)
        target_tokenized = tokenizer(target_sentences)

        return {
            "input_ids": source_tokenized["input_ids"],
            "labels": target_tokenized["input_ids"],
        }
