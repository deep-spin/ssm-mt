from torch.utils.data._utils.collate import default_collate

class SeqBucketCollator:
    def __init__(self, tokenizer, seq_len, label_pad_token_id=None):
        self.tokenizer = tokenizer
        self.seq_len = seq_len
        self.label_pad_token_id = label_pad_token_id

    def __call__(self, batch):
        # Group by sequence length
        buckets = {}
        for item in batch:
            length = len(item["input_ids"])
            bucket_key = min(length // self.seq_len * self.seq_len, self.seq_len)
            if bucket_key not in buckets:
                buckets[bucket_key] = []
            buckets[bucket_key].append(item)

        # Choose the largest bucket
        largest_bucket = max(buckets, key=lambda k: len(buckets[k]))
        bucketed_batch = buckets[largest_bucket]

        # Left-pad the sequences in the bucket
        for item in bucketed_batch:
            padding_length = self.seq_len - len(item["input_ids"])
            item["input_ids"] = [self.tokenizer.pad_token_id] * padding_length + item[
                "input_ids"
            ][: self.seq_len]

        return default_collate(bucketed_batch)
