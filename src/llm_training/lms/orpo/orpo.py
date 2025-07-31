import gc
import logging
from contextlib import nullcontext
from dataclasses import dataclass
from typing import Any

import torch
import torch.nn.functional as F
import torch.distributed.nn.functional as DF
from torch.distributed.tensor import DTensor
from torch.distributed.tensor.parallel import loss_parallel

from llm_training.lightning.strategy.fsdp2.fsdp2_strategy import FSDP2Strategy
from llm_training.lms.base_lm import BaseLightningModule
from llm_training.lms.protos import CausalLMProto
from llm_training.lms.utils import get_model
from llm_training.models.base_model.base_model import BaseModel
from llm_training.ops.cross_entropy_op import shift_labels
from llm_training.ops.liger_kernel import cross_entropy

from .orpo_config import ORPOConfig


@dataclass
class _ForwardOutput:
    chosen_logits: torch.Tensor
    chosen_logps: torch.Tensor
    chosen_labels: torch.Tensor
    rejected_logits: torch.Tensor
    rejected_logps: torch.Tensor
    rejected_labels: torch.Tensor


logger = logging.getLogger(__name__)

class ORPO(BaseLightningModule):
    config: ORPOConfig
    model: CausalLMProto | BaseModel | None

    def __init__(self, config: ORPOConfig) -> None:
        super().__init__(config)

        self.model = None
    
    @property
    def has_pre_trained_weights(self) -> bool:
        return self.model is not None and self.model.has_pre_trained_weights

    def get_pre_trained_weights(self) -> dict[str, torch.Tensor]:
        return {f'model.{k}': v for k, v in self.model.get_pre_trained_weights().items()}
    
    def configure_model(self) -> None:
        self.model = get_model(self.config.model)

        if self.global_rank == 0:
            logger.info(f'Config:\n{repr(self.config)}')
            logger.info(f'Model:\n{self.model}')

    def on_fsdp_parallelize_model(self, **kwargs) -> None:
        self.model.parallelize(**kwargs)

    def get_logps(
        self,
        logits: torch.Tensor | DTensor,
        labels: torch.Tensor
    ) -> torch.Tensor:
        label_mask = labels != self.config.ignore_index
        
        if isinstance(logits, DTensor):
            with loss_parallel():
                logps: DTensor = logits.log_softmax(2)

            local_logps = logps.to_local()
            local_vocab_size = local_logps.size(-1)
            local_vocab_start = logps.device_mesh.get_local_rank() * local_vocab_size
            local_vocab_end = local_vocab_start + local_vocab_size

            local_label_mask = label_mask & (labels >= local_vocab_start) & (labels < local_vocab_end)
            local_index = labels.sub(local_vocab_start).masked_fill_(~local_label_mask, 0).unsqueeze(2)

            per_token_logps = local_logps.gather(2, local_index).squeeze(2)
            per_token_logps[~local_label_mask] = 0.0

            per_token_logps = DF.all_reduce(
                per_token_logps,
                op=torch.distributed.ReduceOp.SUM,
                group=logits.device_mesh.get_group()
            )
        else:
            index = labels.masked_fill(~label_mask, 0).unsqueeze(2)
            per_token_logps = logits.log_softmax(-1).gather(2, index).squeeze(2)
            per_token_logps = per_token_logps * label_mask

        return per_token_logps.sum(-1) / label_mask.sum(-1)

    def forward_batch(self, batch: dict[str, torch.Tensor | Any]) -> _ForwardOutput:
        chosen_labels = shift_labels(batch['chosen_labels'], self.config.ignore_index)
        rejected_labels = shift_labels(batch['rejected_labels'], self.config.ignore_index)

        chosen_logits = self.model(
            input_ids=batch['chosen_input_ids'],
            attention_mask=batch['chosen_attention_mask'],
            position_ids=batch['chosen_position_ids']
        ).logits

        rejected_logits = self.model(
            input_ids=batch['rejected_input_ids'],
            attention_mask=batch['rejected_attention_mask'],
            position_ids=batch['rejected_position_ids']
        ).logits

        chosen_logps = self.get_logps(chosen_logits, chosen_labels)
        rejected_logps = self.get_logps(rejected_logits, rejected_labels)

        return _ForwardOutput(
            chosen_logits=chosen_logits,
            chosen_logps=chosen_logps,
            chosen_labels=chosen_labels,
            rejected_logits=rejected_logits,
            rejected_logps=rejected_logps,
            rejected_labels=rejected_labels
        )
    
    def compute_loss(
        self,
        chosen_logits: torch.Tensor,
        chosen_logps: torch.Tensor,
        chosen_labels: torch.Tensor,
        rejected_logits: torch.Tensor,
        rejected_logps: torch.Tensor,
        rejected_labels: torch.Tensor,
        metrics: dict[str, torch.Tensor]
    ) -> torch.Tensor:
        beta = self.config.beta
        log_odds = (chosen_logps - rejected_logps) - (
            torch.log1p(-torch.exp(chosen_logps)) - torch.log1p(-torch.exp(rejected_logps))
        )
        ratio = F.logsigmoid(log_odds)
        or_loss = -(beta * ratio).mean()
        
        metrics['OR Loss'] = or_loss
        chosen_rewards = beta * chosen_logps.detach()
        rejected_rewards = beta * rejected_logps.detach()
        metrics['Chosen Rewards'] = chosen_rewards.mean()
        metrics['Rejected Rewards'] = rejected_rewards.mean()
        metrics['Reward Accuracy'] = (chosen_rewards > rejected_rewards).float().mean()
        metrics['Reward Margin'] = (chosen_rewards - rejected_rewards).mean()
        metrics['Chosen Log P'] = chosen_logps.detach().mean()
        metrics['Rejected Log P'] = rejected_logps.detach().mean()
        metrics['Chosen Logits'] = chosen_logits.detach().mean()
        metrics['Rejected Logits'] = rejected_logits.detach().mean()
        metrics['Log Odds Ratio'] = ratio.detach().mean()
        metrics['Log Odds Chosen'] = log_odds.detach().mean()

        if isinstance(self.strategy, FSDP2Strategy) and self.strategy.tp_size > 1:
            with loss_parallel():
                ce_loss = F.cross_entropy(
                    chosen_logits.flatten(end_dim=1),
                    chosen_labels.flatten(end_dim=1),
                    ignore_index=self.config.ignore_index
                )
        else:
            ce_loss = cross_entropy(
                chosen_logits,
                chosen_labels,
                self.config.ignore_index
            )
        
        metrics['CE Loss'] = ce_loss

        if isinstance(ce_loss, DTensor):
            or_loss = DTensor.from_local(
                or_loss,
                ce_loss.device_mesh,
                ce_loss.placements,
                run_check=False
            )

        return or_loss + ce_loss
    
    def backward(self, loss: torch.Tensor, *args, **kwargs) -> None:
        backward_ctx = nullcontext()
        if isinstance(self.strategy, FSDP2Strategy) and self.strategy.tp_size > 1:
            backward_ctx = loss_parallel()

        with backward_ctx:
            return super().backward(loss, *args, **kwargs)
    
    def add_suffix_to_metrics(self, metrics: dict[str, torch.Tensor], suffix: str) -> dict[str, torch.Tensor]:
        return {k + suffix: v for k, v in metrics.items()}

    def training_step(self, batch: dict[str, torch.Tensor | Any], batch_idx: int) -> torch.Tensor:
        max_seq_len = max(batch['chosen_input_ids'].size(1), batch['rejected_input_ids'].size(1))
        if (
            self.config.empty_cache_threshold is not None
            and max_seq_len >= self.config.empty_cache_threshold
        ):
            gc.collect()
            torch.cuda.empty_cache()

        outputs = self.forward_batch(batch)

        metrics = {}
        loss = self.compute_loss(
            chosen_logits=outputs.chosen_logits,
            chosen_logps=outputs.chosen_logps,
            chosen_labels=outputs.chosen_labels,
            rejected_logits=outputs.rejected_logits,
            rejected_logps=outputs.rejected_logps,
            rejected_labels=outputs.rejected_labels,
            metrics=metrics
        )
        metrics['Loss'] = loss

        self.log('loss', loss, prog_bar=True, logger=False)

        metrics = self.add_suffix_to_metrics(metrics, '/Train/Step')
        self.log_dict(metrics)
        return loss

    def validation_step(self, batch: dict[str, torch.Tensor | Any], batch_idx: int) -> None:
        batch_size = batch['chosen_input_ids'].size(0)

        outputs = self.forward_batch(batch)

        metrics = {}
        loss = self.compute_loss(
            chosen_logits=outputs.chosen_logits,
            chosen_logps=outputs.chosen_logps,
            chosen_labels=outputs.chosen_labels,
            rejected_logits=outputs.rejected_logits,
            rejected_logps=outputs.rejected_logps,
            rejected_labels=outputs.rejected_labels,
            metrics=metrics
        )
        metrics['Loss'] = loss

        for k, v in metrics.items():
            if isinstance(v, DTensor):
                metrics[k] = v.full_tensor()
        
        metrics = self.add_suffix_to_metrics(metrics, '/Val')
        self.log_dict(metrics, sync_dist=True, batch_size=batch_size)

    def get_model(self) -> BaseModel:
        return self.model
