"""GMM VAE for NLP."""

from typing import Tuple, Dict
from dataclasses import dataclass

import torch
import pytorch_lightning as pl
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau

@dataclass
class Config:

    vocab_size: int
    hidden_dim: int
    z_dim: int
    pad_idx: int = 0
    unk_idx: int = 1
    bos_idx: int = 2
    eos_idx: int = 3
    max_sent_len: int = 15
    learning_rate: float = 2e-3


class GMMVAE(pl.LightningModule):

    def __init__(self, config: Config, idx2token: Dict[int, str] = None) -> None:
        """
        Variational Autoencoder with Gaussian Mixture Model.

        ┌──────┐                                        ┌──────┐
        │      │                                        │      │
        │      │              ┌────┐                    │      │
        │      │              │    │                    │      │
        │      │  ──────────► │ μ  ├────────┐           │      │
        │      │              │    │        ▼           │      │
        │      │              │    │     ┌─────┐        │      │
        │      │              └────┘     │     │        │      │
        │      │                         │     │        │      │
        │      │                         │  z  │ ─────► │   x' │
        │   x  │                         │     │        │      │
        │      │              ┌────┐     │     │        │      │
        │      │              │    │     └─────┘        │      │
        │      │              │ σ  │        ▲           │      │
        │      │  ──────────► │    ├────────┘           │      │
        │      │              │    │                    │      │
        │      │              └────┘                    │      │
        │      │                                        │      │
        │      │                                        │      │
        └──────┘                                        └──────┘

        VAE assumes that a document/input/example/sentence follow a certain prior distribution 
        and its goal is to approximate such distribution by learning the distribution mean μ and variance σ^2. 
        When you have a mean μ and a variance σ^2, you can *sample a distribution ~ N(μ, σ^2)* (z in this case) to 
        represent the document's topic distribution. 

        If we assume that z ~ N(0, 1) normal distribution, then our goal includes:
        - reconstruction of x given such distribution z – construction loss
        - such distribution z should align with the normal distribution - KL divergence loss

        Parameters
        ----------
        config : Config
            Model configurations
        """
        super().__init__()
        self.cfg = config
        self.idx2token = idx2token

        self.embedder = nn.Embedding(self.cfg.vocab_size, self.cfg.hidden_dim, padding_idx=self.cfg.pad_idx)

        self.encoder = nn.GRU(self.cfg.hidden_dim, self.cfg.hidden_dim, batch_first=True)
        
        self.hidden2mu = nn.Linear(self.cfg.hidden_dim, self.cfg.z_dim)
        # We do not model the variance σ^2 directly. 
        # Instead, we model the log variance log(σ^2).
        self.hidden2log_variance = nn.Linear(self.cfg.hidden_dim, self.cfg.z_dim)
        self.decoder = nn.GRU(self.cfg.hidden_dim + self.cfg.z_dim, self.cfg.z_dim, batch_first=True)
        self.decoder_fc = nn.Linear(self.cfg.z_dim, self.cfg.vocab_size)


    def encode(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:

        embeddings = self.embedder(x)
        _, hidden = self.encoder(embeddings)
        # hidden: [# layers, sequence, hidden]
        mu = self.hidden2mu(hidden)
        log_variance = self.hidden2log_variance(hidden)

        return mu, log_variance
    
    def forward(self, input_ids: torch.Tensor, decoder_input_ids: torch.Tensor) -> torch.Tensor:
        
        mu, log_variance = self.encode(input_ids)
        z = self.reparametrized_sampling(mu, log_variance)
        y = self.decode(decoder_input_ids, z)
        
        return mu, log_variance, y
    
    def decode(self, decoder_input_ids: torch.Tensor, z: torch.Tensor) -> torch.Tensor:
        
        embeddings = self.embedder(decoder_input_ids)
        embeddings = torch.cat([embeddings, z.squeeze(0).unsqueeze(1).repeat(1, embeddings.shape[1], 1)], dim=-1)
        outputs, _ = self.decoder(embeddings, z)
        y = self.decoder_fc(outputs)

        return y


    def training_step(self, batch, batch_idx=None):
        
        input_ids = batch["input_ids"]
        decoder_input_ids = batch["decoder_input_ids"]
        decoder_target_ids = batch["decoder_target_ids"]

        mu, log_variance, outputs = self.forward(input_ids, decoder_input_ids)

        kl_loss = torch.mean(0.5 * torch.sum(torch.exp(log_variance) + mu ** 2 - 1 - log_variance, 1))
        recon_loss = F.cross_entropy(
            outputs.view(-1, self.cfg.vocab_size), decoder_target_ids.view(-1), reduction='mean'
        )
        loss = recon_loss + kl_loss

        self.log('train_loss', loss, on_step=False, on_epoch=True, prog_bar=True)
        
        return loss

    def validation_step(self, batch, batch_idx=None):
        
        input_ids = batch["input_ids"]
        decoder_input_ids = batch["decoder_input_ids"]
        decoder_target_ids = batch["decoder_target_ids"]

        mu, log_variance, outputs = self.forward(input_ids, decoder_input_ids)

        kl_loss = torch.mean(0.5 * torch.sum(torch.exp(log_variance) + mu ** 2 - 1 - log_variance, 1))
        recon_loss = F.cross_entropy(
            outputs.view(-1, self.cfg.vocab_size), decoder_target_ids.view(-1), reduction='mean'
        )
        loss = recon_loss + kl_loss
        self.log('val_loss', loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log('val_kl_loss', kl_loss, on_step=False, on_epoch=True)
        self.log('val_recon_loss', recon_loss, on_step=False, on_epoch=True)

        return loss

    def validation_epoch_end(self, outputs):
        
        sentence = self.generate_samples(1.0)
        
        print(" ".join(self.idx2token.get(idx, None) for idx in sentence))

    def configure_optimizers(self):
        optimizer = Adam(self.parameters(), lr=self.cfg.learning_rate)
        lr_scheduler = ReduceLROnPlateau(optimizer)
        return {
            "optimizer": optimizer, "lr_scheduler": lr_scheduler,
            "monitor": "val_loss"
        }

    @staticmethod
    def reparametrized_sampling(mu: torch.Tensor, log_variance: torch.Tensor) -> torch.Tensor:
        """
        Usually, when you sample a probability distribution given mean μ and variance σ,
        it is not differentiable and therefore not trainable from the neural network perspective.
        This is a reparameterization trick so that the sampling process can allow gradients to 
        backpropagate.

        Parameters
        ----------
        mu : torch.Tensor
            Predicted input mean, with a shape [batch, hidden]
        log_variance : torch.Tensor
            Predicted input log variance log(σ^2), with a shape [batch, hidden]

        Returns
        -------
        torch.Tensor
            Sampled z, aka topic distribution
        """

        # log(σ^2) -> e^(1/2 * log(σ^2)) -> σ
        sigma = torch.exp(0.5*log_variance)
        epsilon = torch.randn(mu.size(), dtype=mu.dtype, device=mu.device)
        return mu + sigma * epsilon


    def generate_samples(self, temperature=1.0):

        # Sampling a topic probability distribution for generation
        z = torch.randn(1, self.cfg.z_dim)

        with torch.no_grad():

            word = torch.LongTensor([self.cfg.bos_idx])
            z = z.view(1, 1, -1)
            h = z

            results = [self.cfg.bos_idx]

            for _ in range(self.cfg.max_sent_len):

                embeddings = self.embedder(word).view(1, 1, -1)
                embeddings = torch.cat([embeddings, z.squeeze(0).unsqueeze(1).repeat(1, embeddings.shape[1], 1)], dim=-1)
                outputs, h = self.decoder(embeddings, h)
                y = self.decoder_fc(outputs).view(-1)
                y = F.softmax(y/temperature, dim=0)
                idx = torch.multinomial(y, 1)
                word = torch.LongTensor([idx])
                idx = int(idx)

                if idx == self.cfg.eos_idx:
                    break

                results.append(idx)
        
        return results