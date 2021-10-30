
from model import GMMVAE, Config
from dataset import SSTDataModule

import pytorch_lightning as pl

dataset = SSTDataModule()
config = Config(len(dataset.token2idx), 128, 32)
model = GMMVAE(config, dataset.idx2token)

trainer = pl.Trainer()
trainer.fit(model, dataset)