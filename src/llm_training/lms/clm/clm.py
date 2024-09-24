import logging
from typing import Any

import torch
from torch import nn
from torchmetrics.text import Perplexity

from llm_training.lms.base_lm import BaseLightningModule
from llm_training.lms.protos import CausalLMProto
from llm_training.lms.utils import get_model
from llm_training.metrics import ConsumedSamples, ConsumedTokens
from llm_training.models.base_model.base_model import BaseModel
from llm_training.ops import shift_labels
from llm_training.ops.liger import cross_entropy

from .clm_config import CLMConfig

logger = logging.getLogger(__name__)


class CLM(BaseLightningModule):
    config: CLMConfig
    model: CausalLMProto | BaseModel | None

    def __init__(self, config: CLMConfig) -> None:
        super().__init__(config)

        self.model = None
        self.train_perplexity = Perplexity(ignore_index=self.config.ignore_index)
        self.val_perplexity = Perplexity(ignore_index=self.config.ignore_index)
        self.consumed_samples = ConsumedSamples()
        self.consumed_tokens = ConsumedTokens(ignore_index=self.config.ignore_index)
    
    @property
    def has_pre_trained_weights(self) -> bool:
        if self.model is None:
            return False
        return self.model.has_pre_trained_weights

    def get_pre_trained_weights(self) -> dict[str, torch.Tensor]:
        state_dict = self.model.get_pre_trained_weights()
        state_dict = {f'model.{k}': v for k, v in state_dict.items()}
        return state_dict

    def neftune_forward_hook(
        self,
        module: nn.Module,
        input: torch.Tensor,
        output: torch.Tensor
    ) -> torch.Tensor:
        if module.training:
            attention_mask = getattr(self, '_current_attention_mask', None)
            if attention_mask is None:
                attention_mask = torch.ones_like(input)
                
            # packed attention mask
            attention_mask = attention_mask.bool().to(output.dtype)

            noise = torch.zeros_like(output).uniform_(-1, 1)
            input_lengths = attention_mask.sum(1)
            delta = noise * attention_mask.unsqueeze(2)
            dims = input_lengths * output.size(-1)
            magnitude = self.config.neftune_alpha / torch.sqrt(dims)
            delta = (delta * magnitude.view(-1, 1, 1)).detach()
            output = output + delta
        return output
    
    def register_neftune_hook(self) -> None:
        embedding = self.model.get_input_embeddings()
        self._neftune_hook_handle = embedding.register_forward_hook(self.neftune_forward_hook)

    def configure_model(self) -> None:
        self.model = get_model(self.config.model)

        if self.global_rank == 0:
            logger.info(f'Config:\n{repr(self.config)}')
            logger.info(f'Model:\n{self.model}')

        if self.config.neftune_alpha is not None:
            self.register_neftune_hook()

    def on_fsdp_wrap_model(self, state_dict: dict[str, torch.Tensor] | None) -> None:
        assert self.model.no_split_modules
        self.model = self.fsdp_wrap_model(self.model, 'model', state_dict, self.model.no_split_modules)

    def compute_loss(self, logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        return cross_entropy(
            logits,
            labels,
            ignore_index=self.config.ignore_index,
            reduction='mean'
        )

    def forward_batch(self, batch: dict[str, torch.Tensor | Any]) -> tuple[torch.Tensor, torch.Tensor]:
        kwargs = dict(
            input_ids=batch['input_ids'],
            attention_mask=batch['attention_mask']
        )

        if 'position_ids' in batch:
            kwargs['position_ids'] = batch['position_ids']

        labels = batch['labels']
        if 'pixel_values' in batch:
            kwargs['labels'] = labels
            kwargs['pixel_values'] = batch['pixel_values']
            attention_mask, labels, logits = self.model(**kwargs)
            attention_mask = shift_labels(attention_mask, 0)
            labels = shift_labels(labels, self.config.ignore_index)
            labels[attention_mask == 0] = self.config.ignore_index
        else:
            logits = self.model(**kwargs)
            labels = shift_labels(labels, self.config.ignore_index)

        return logits, labels

    def training_step(self, batch: dict[str, torch.Tensor | Any], batch_idx: int) -> torch.Tensor:
        if self.config.neftune_alpha is not None:
            self._current_attention_mask = batch['attention_mask']

        logits, labels = self.forward_batch(batch)

        if self.config.neftune_alpha is not None:
            self.log('NEFTune Alpha', self.config.neftune_alpha)
            self._current_attention_mask = None

        # compute CE loss after ppl, because some CE kernels lead to wrong ppl.
        self.train_perplexity(logits, labels)
        loss = self.compute_loss(logits, labels)

        self.log('loss', loss, prog_bar=True, logger=False)
        self.log('Loss/Train/Step', loss)
        self.log('Perplexity/Train/Step', self.train_perplexity)

        if self.grad_norm is not None:
            self.log('Gradient Norm', self.grad_norm)

        self.consumed_samples.update(labels)
        self.consumed_tokens.update(labels)
        self.logger.log_metrics({
            'Consumed Samples': self.consumed_samples.compute(),
            'Consumed Tokens': self.consumed_tokens.compute()
        })
        return loss

    def validation_step(self, batch: dict[str, torch.Tensor | Any], batch_idx: int, dataloader_idx: int = 0):
        batch_size = batch['input_ids'].size(0)
        logits, labels = self.forward_batch(batch)

        self.val_perplexity.update(logits, labels)
        loss = self.compute_loss(logits, labels)

        self.log('Loss/Val', loss, batch_size=batch_size, sync_dist=True)
        self.log('Perplexity/Val', self.val_perplexity)

    def get_model(self) -> BaseModel:
        return self.model
