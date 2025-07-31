import logging
from contextlib import nullcontext
from typing import Any

import torch
import torch.nn.functional as F
from torch import nn
from torch.distributed.tensor import DTensor
from torch.distributed.tensor.parallel import loss_parallel

from llm_training.lightning.strategy import FSDP2Strategy
from llm_training.lms.base_lm import BaseLightningModule
from llm_training.lms.protos import CausalLMProto
from llm_training.lms.utils import get_model
from llm_training.metrics import ConsumedSamples, ConsumedTokens, Perplexity
from llm_training.models.base_model.base_model import BaseModel
from llm_training.ops import shift_labels
from llm_training.ops.liger_kernel import cross_entropy

from .clm_config import CLMConfig

logger = logging.getLogger(__name__)


class CLM(BaseLightningModule):
    config: CLMConfig
    model: CausalLMProto | BaseModel | None

    def __init__(self, config: CLMConfig) -> None:
        super().__init__(config)

        self.model = None
    
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
                
            # For packed attention mask
            attention_mask = attention_mask.bool().to(output.dtype)
            
            noise = torch.empty(
                output.shape,
                dtype=output.dtype,
                device=output.device
            )
            noise = noise.uniform_(-1, 1)
            input_lengths = attention_mask.sum(1)
            delta = noise * attention_mask.unsqueeze(2)
            dims = input_lengths * output.size(-1)
            magnitude = self.config.neftune_alpha / torch.sqrt(dims)
            delta = (delta * magnitude.view(-1, 1, 1)).detach()
            if isinstance(output, DTensor):
                delta = DTensor.from_local(
                    delta,
                    device_mesh=output.device_mesh,
                    placements=output.placements,
                    run_check=False
                )
            output = output + delta
        return output
    
    def register_neftune_hook(self) -> None:
        embedding = self.model.get_input_embeddings()
        self._neftune_hook_handle = embedding.register_forward_hook(self.neftune_forward_hook)

    def configure_model(self) -> None:
        process_group = self.strategy.dp_mesh.get_group() if isinstance(self.strategy, FSDP2Strategy) else None
        self.consumed_samples = ConsumedSamples(process_group=process_group)
        self.consumed_tokens = ConsumedTokens(
            ignore_index=self.config.ignore_index,
            process_group=process_group
        )
        if self.config.log_perplexity:
            self.train_perplexity = Perplexity(
                ignore_index=self.config.ignore_index,
                process_group=process_group
            )
            self.val_perplexity = Perplexity(
                ignore_index=self.config.ignore_index,
                process_group=process_group
            )

        self.model = get_model(self.config.model)

        if self.global_rank == 0:
            logger.info(f'Config:\n{repr(self.config)}')
            logger.info(f'Model:\n{self.model}')

        if self.config.neftune_alpha is not None:
            self.register_neftune_hook()

    def on_fsdp_parallelize_model(self, **kwargs) -> None:
        self.model.parallelize(**kwargs)

    def compute_loss(self, logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        if isinstance(self.strategy, FSDP2Strategy) and self.strategy.tp_size > 1:
            with loss_parallel():
                return F.cross_entropy(
                    logits.flatten(end_dim=1),
                    labels.flatten(end_dim=1),
                    ignore_index=self.config.ignore_index
                )
        
        return cross_entropy(
            logits=logits,
            labels=labels,
            ignore_index=self.config.ignore_index
        )

    def backward(self, loss: torch.Tensor, *args, **kwargs) -> None:
        backward_ctx = nullcontext()
        if isinstance(self.strategy, FSDP2Strategy) and self.strategy.tp_size > 1:
            backward_ctx = loss_parallel()

        with backward_ctx:
            super().backward(loss, *args, **kwargs)

    def training_step(self, batch: dict[str, torch.Tensor | Any], batch_idx: int) -> torch.Tensor:
        labels = shift_labels(batch['labels'], self.config.ignore_index)
        
        if self.config.neftune_alpha is not None:
            self._current_attention_mask = batch['attention_mask']

        outputs = self.model(
            input_ids=batch['input_ids'],
            attention_mask=batch['attention_mask'],
            position_ids=batch.get('position_ids', None)
        )
        logits = outputs.logits.float()

        if self.config.neftune_alpha is not None:
            self.log('NEFTune Alpha', self.config.neftune_alpha)
            self._current_attention_mask = None
        
        loss = self.compute_loss(logits, labels)

        self.log('loss', loss, prog_bar=True, logger=False)
        self.log('Loss/Train/Step', loss)

        if self.config.log_perplexity:
            self.train_perplexity(loss)
            self.log('Perplexity/Train/Step', self.train_perplexity)

        self.consumed_samples.update(labels)
        self.consumed_tokens.update(labels)
        self.logger.log_metrics({
            'Consumed Samples': self.consumed_samples.compute(),
            'Consumed Tokens': self.consumed_tokens.compute()
        })
        return loss

    def validation_step(self, batch: dict[str, torch.Tensor | Any], batch_idx: int, dataloader_idx: int = 0):
        batch_size = batch['input_ids'].size(0)
        labels = shift_labels(batch['labels'], self.config.ignore_index)
        outputs = self.model(
            input_ids=batch['input_ids'],
            attention_mask=batch['attention_mask'],
            position_ids=batch.get('position_ids', None)
        )
        logits = outputs.logits.float()

        loss = self.compute_loss(logits, labels)

        if isinstance(loss, DTensor):
            loss = loss.full_tensor()
        
        self.log('Loss/Val', loss, batch_size=batch_size, sync_dist=True)

        if self.config.log_perplexity:
            self.val_perplexity.update(loss)
            self.log('Perplexity/Val', self.val_perplexity)

    def get_model(self) -> BaseModel:
        return self.model
