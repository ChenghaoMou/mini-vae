
from typing import Optional
from itertools import chain

import torch
import numpy as np
import pytorch_lightning as pl
from torch.utils.data import DataLoader, Dataset
from datasets import load_dataset
from sklearn.model_selection import train_test_split as tts

class VAEDataset(Dataset):

    def __init__(self, input_ids, labels):
        self.input_ids = input_ids
        self.labels = labels
    
    def __len__(self):
        return len(self.input_ids)
    
    def __getitem__(self, idx):
        return {
            "input_ids": self.input_ids[idx],
            "decoder_input_ids": self.input_ids[idx][:-1],
            "decoder_target_ids": self.input_ids[idx][1:],
            "labels": self.labels[idx]
        }

class SSTDataModule(pl.LightningDataModule):
    def __init__(self, batch_size: int = 32):
        super().__init__()
        self.batch_size = batch_size
        self.token2idx = {"PAD": 0, "UNK": 1, "BOS": 2, "EOS": 3,}
        self.idx2token = {idx: token for token, idx in self.token2idx.items()}

        dataset = load_dataset("sst", "default")
        X = []
        Y = []
        for record in chain(dataset["train"], dataset["validation"]):
            tokens = record["tokens"].split('|')
            
            if len(tokens) > 15:
                continue

            for token in tokens:
                if token.lower() not in self.token2idx:
                    self.token2idx[token.lower()] = len(self.token2idx)
                    self.idx2token[self.token2idx[token.lower()]] = token.lower()
            
            X.append([2] + [self.token2idx[token.lower()] for token in tokens] + [3])
            Y.append(record["label"])
        
        train_X, val_X, train_Y, val_Y = tts(X, Y, test_size=0.3)
        self.train = VAEDataset(train_X, train_Y)
        self.val = VAEDataset(val_X, val_Y)

    def setup(self, stage: Optional[str] = None):
        
        pass

    def train_dataloader(self):
        return DataLoader(self.train, batch_size=self.batch_size, collate_fn=self.collate_fn)

    def val_dataloader(self):
        return DataLoader(self.val, batch_size=self.batch_size, collate_fn=self.collate_fn)

    def collate_fn(self, examples):

        input_ids = np.zeros((len(examples), max(map(lambda x: len(x["input_ids"]), examples))))
        decoder_input_ids = np.zeros((len(examples), max(map(lambda x: len(x["decoder_input_ids"]), examples))))
        decoder_target_ids = np.zeros((len(examples), max(map(lambda x: len(x["decoder_target_ids"]), examples))))
        labels = np.zeros((len(examples)))

        for i, x in enumerate(examples):
            input_ids[i][:len(x["input_ids"])] = x["input_ids"]
            decoder_input_ids[i][:len(x["decoder_input_ids"])] = x["decoder_input_ids"]
            decoder_target_ids[i][:len(x["decoder_target_ids"])] = x["decoder_target_ids"]
            labels[i] = x["labels"]
        
        return {
            "input_ids": torch.from_numpy(input_ids).long(),
            "decoder_input_ids": torch.from_numpy(decoder_input_ids).long(),
            "decoder_target_ids": torch.from_numpy(decoder_target_ids).long(),
            "labels": torch.from_numpy(labels).float()

        }





