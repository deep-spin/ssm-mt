import os
from datasets import load_dataset, DatasetDict, Dataset
from transformers import (
    PreTrainedTokenizerFast,
    DataCollatorForSeq2Seq,
    DataCollatorWithPadding,
)


from utils.mt import DataCollatorDecForSeq2Seq


from mt.ds.base import BaseDataset


class WMT23Dataset(BaseDataset):
    name: str = "wmt23"
    dataset: DatasetDict = None
    source_lang: str = None
    target_lang: str = None
    is_encoder_decoder: bool = False

    def __init__(self, source, target, is_encoder_decoder):
        self.is_encoder_decoder = is_encoder_decoder
        self.source_lang, self.target_lang = (source, target)
        self.dataset = DatasetDict.load_from_disk(f"data/wmt23/{source}-{target}")

    def get_ckpt_path(self, model_name):
        return os.path.join(
            f"data/mt/{self.name}-{model_name}",
            f"{self.source_lang}-{self.target_lang}",
        )

    def get_tokenizer(self) -> PreTrainedTokenizerFast:
        source, target = (
            (self.source_lang, self.target_lang)
            if self.source_lang == "en"
            else (self.target_lang, self.source_lang)
        )

        # this is for the models trained with the news tokenizer
        dir = os.path.join("data/mt/tokenizers", f"news-{source}-{target}")
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

    def get_data_iterator(self):
        for i in range(0, len(self.dataset["train"])):

            yield self.dataset["train"][i][self.source_lang] + " " + self.dataset[
                "train"
            ][i][self.target_lang]

        for i in range(0, len(self.dataset["val"])):
            yield self.dataset["val"][i][self.source_lang] + " " + self.dataset["val"][
                i
            ][self.target_lang]

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
            train_columns = val_columns = [
                "input_ids",
                "labels",
            ]
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
                remove_columns=["en", "de"],
                collate_fn=t_collate_fn,
            )

            val_dl = self.get_dataloader(
                self.dataset["validation"],
                fn=val_fn,
                fn_kwargs=val_fn_kwargs,
                batch_size=val_batch_size,
                columns=val_columns,
                remove_columns=["en", "de"],
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
                remove_columns=["en", "de"],
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
            f"{sample[0]}{tokenizer.sep_token}{sample[1]}"
            for sample in zip(batch[self.source_lang], batch[self.target_lang])
        ]

        source_tokenized = tokenizer.batch_encode_plus(
            source_sentences, truncation=True, max_length=1024
        )

        return {
            "input_ids": source_tokenized["input_ids"],
            "attention_mask": source_tokenized["attention_mask"],
        }

    def encdec_training_preprocess(
        self,
        batch,
        tokenizer: PreTrainedTokenizerFast,
    ):
        source_sentences = [sample for sample in batch[self.source_lang]]
        target_sentences = [
            tokenizer.bos_token + sample for sample in batch[self.target_lang]
        ]

        source_tokenized = tokenizer(source_sentences, truncation=True, max_length=768)
        target_tokenized = tokenizer(target_sentences, truncation=True, max_length=768)
        # invert the mask
        return {
            "input_ids": source_tokenized["input_ids"],
            "attention_mask": source_tokenized["attention_mask"],
            "labels": target_tokenized["input_ids"],
        }

    def validation_preprocess(self, batch, tokenizer: PreTrainedTokenizerFast):
        source_sentences = [
            f"{sample}{tokenizer.sep_token}" for sample in batch[self.source_lang]
        ]

        target_sentences = [f"{sample}" for sample in batch[self.target_lang]]

        source_tokenized = tokenizer.batch_encode_plus(
            source_sentences, truncation=True, max_length=1024
        )
        target_tokenized = tokenizer.batch_encode_plus(
            target_sentences, truncation=True, max_length=1024
        )

        input_ids = [sample[:-1] for sample in source_tokenized["input_ids"]]
        attention_mask = [sample[:-1] for sample in source_tokenized["attention_mask"]]

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": target_tokenized["input_ids"],
        }

    def encdec_validation_preprocess(
        self,
        batch,
        tokenizer: PreTrainedTokenizerFast,
    ):
        source_sentences = [sample for sample in batch[self.source_lang]]
        target_sentences = [
            tokenizer.bos_token + sample for sample in batch[self.target_lang]
        ]

        source_tokenized = tokenizer(source_sentences, truncation=True, max_length=768)
        target_tokenized = tokenizer(target_sentences, truncation=True, max_length=768)
        # invert the mask
        return {
            "input_ids": source_tokenized["input_ids"],
            "attention_mask": source_tokenized["attention_mask"],
            "labels": target_tokenized["input_ids"],
        }

    def test_preprocess(self, batch, tokenizer: PreTrainedTokenizerFast):
        pass
