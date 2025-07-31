import logging
from contextlib import nullcontext
from dataclasses import dataclass
from typing import Any

import torch
import torch.nn.functional as F
import torch.distributed.nn.functional as DF
from torch.distributed.tensor import DTensor
from torch.distributed.tensor.parallel import loss_parallel

from llm_training.lightning import FSDP2Strategy
from llm_training.lms.base_lm import BaseLightningModule
from llm_training.lms.protos import CausalLMProto
from llm_training.lms.utils import get_model
from llm_training.models.base_model.base_model import BaseModel
from llm_training.ops.cross_entropy_op import shift_labels

from .dpo_config import DPOConfig

logger = logging.getLogger(__name__)

@dataclass
class _ForwardOutput:
    policy_chosen_logps: torch.Tensor
    policy_rejected_logps: torch.Tensor
    reference_chosen_logps: torch.Tensor
    reference_rejected_logps: torch.Tensor


class DPO(BaseLightningModule):
    config: DPOConfig
    model: CausalLMProto | BaseModel | None
    ref_model: CausalLMProto | BaseModel | None


    def __init__(self, config: DPOConfig) -> None:
        super().__init__(config)

        self.model = None
        self.ref_model = None
    
    @property
    def has_pre_trained_weights(self) -> bool:
        if self.model is None:
            return False
        return self.model.has_pre_trained_weights

    def get_pre_trained_weights(self) -> dict[str, torch.Tensor]:
        model_state_dict = self.model.get_pre_trained_weights()
        ref_model_state_dict = (
            model_state_dict if self.config.ref_model is None
            else self.ref_model.get_pre_trained_weights()
        )
        state_dict = {}
        state_dict |= {f'model.{k}': v for k, v in model_state_dict.items()}
        state_dict |= {f'ref_model.{k}': v for k, v in ref_model_state_dict.items()}
        return state_dict
    
    def configure_model(self) -> None:
        self.model = get_model(self.config.model)
        self.ref_model = get_model(self.config.ref_model or self.config.model)
        self.ref_model.eval().requires_grad_(False)

        if self.global_rank == 0:
            logger.info(f'Config:\n{repr(self.config)}')
            logger.info(f'Model:\n{self.model}')
            logger.info(f'Reference Model:\n{self.ref_model}')

    def on_fsdp_parallelize_model(self, **kwargs) -> None:
        self.model.parallelize(**kwargs)
        self.ref_model.parallelize(**kwargs)

    def get_logps(
        self,
        model: CausalLMProto,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        position_ids: torch.Tensor,
        labels: torch.Tensor
    ) -> torch.Tensor:
        label_mask = labels != self.config.ignore_index
        
        logits = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids
        ).logits

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
        
        return per_token_logps.sum(-1)

    def forward_batch(self, batch: dict[str, torch.Tensor | Any]) -> _ForwardOutput:
        chosen_labels = shift_labels(batch['chosen_labels'], self.config.ignore_index)
        rejected_labels = shift_labels(batch['rejected_labels'], self.config.ignore_index)
    
        policy_chosen_logps = self.get_logps(
            self.model,
            batch['chosen_input_ids'],
            batch['chosen_attention_mask'],
            batch['chosen_position_ids'],
            chosen_labels
        )
        policy_rejected_logps = self.get_logps(
            self.model,
            batch['rejected_input_ids'],
            batch['rejected_attention_mask'],
            batch['rejected_position_ids'],
            rejected_labels
        )
        reference_chosen_logps = self.get_logps(
            self.ref_model,
            batch['chosen_input_ids'],
            batch['chosen_attention_mask'],
            batch['chosen_position_ids'],
            batch['chosen_labels']
        )
        reference_rejected_logps = self.get_logps(
            self.ref_model,
            batch['rejected_input_ids'],
            batch['rejected_attention_mask'],
            batch['rejected_position_ids'],
            batch['rejected_labels']
        )

        return _ForwardOutput(
            policy_chosen_logps=policy_chosen_logps,
            policy_rejected_logps=policy_rejected_logps,
            reference_chosen_logps=reference_chosen_logps,
            reference_rejected_logps=reference_rejected_logps
        )
    
    def compute_loss(
        self,
        policy_chosen_logps: torch.Tensor,
        policy_rejected_logps: torch.Tensor,
        reference_chosen_logps: torch.Tensor,
        reference_rejected_logps: torch.Tensor,
        metrics: dict[str, torch.Tensor]
    ) -> torch.Tensor:
        pi_logratios = policy_chosen_logps - policy_rejected_logps
        ref_logratios = reference_chosen_logps - reference_rejected_logps
        logits = pi_logratios - ref_logratios
        loss = (
            -F.logsigmoid(self.config.beta * logits) * (1 - self.config.label_smoothing)
            - F.logsigmoid(-self.config.beta * logits) * self.config.label_smoothing
        )
        loss = loss.mean()

        chosen_rewards = self.config.beta * (policy_chosen_logps - reference_chosen_logps).detach()
        rejected_rewards = self.config.beta * (policy_rejected_logps - reference_rejected_logps).detach()
        reward_accuracy = (chosen_rewards > rejected_rewards).float().mean()
        chosen_reward = chosen_rewards.mean()
        rejected_reward = rejected_rewards.mean()
        reward_margin = (chosen_rewards - rejected_rewards).mean()
        
        metrics['Chosen Reward'] = chosen_reward
        metrics['Rejected Reward'] = rejected_reward
        metrics['Reward Accuracy'] = reward_accuracy
        metrics['Reward Margin'] = reward_margin
        metrics['Chosen Log P'] = policy_chosen_logps.mean()
        metrics['Rejected Log P'] = policy_rejected_logps.mean()

        return loss

    def backward(self, loss: torch.Tensor, *args, **kwargs) -> None:
        backward_ctx = nullcontext()
        if isinstance(self.strategy, FSDP2Strategy) and self.strategy.tp_size > 1:
            backward_ctx = loss_parallel()

        with backward_ctx:
            return super().backward(loss, *args, **kwargs)

    def add_suffix_to_metrics(self, metrics: dict[str, torch.Tensor], suffix: str) -> dict[str, torch.Tensor]:
        return {k + suffix: v for k, v in metrics.items()}

    def training_step(self, batch: dict[str, torch.Tensor | Any], batch_idx: int) -> torch.Tensor:
        output = self.forward_batch(batch)

        metrics = {}
        loss = self.compute_loss(
            policy_chosen_logps=output.policy_chosen_logps,
            policy_rejected_logps=output.policy_rejected_logps,
            reference_chosen_logps=output.reference_chosen_logps,
            reference_rejected_logps=output.reference_rejected_logps,
            metrics=metrics
        )
        metrics['Loss'] = loss

        self.log('loss', loss, prog_bar=True, logger=False)
        
        metrics = self.add_suffix_to_metrics(metrics, '/Train/Step')
        self.log_dict(metrics)
        return loss

    def validation_step(self, batch: dict[str, torch.Tensor | Any], batch_idx: int) -> None:
        batch_size = batch['chosen_input_ids'].size(0)

        output = self.forward_batch(batch)

        metrics = {}
        loss = self.compute_loss(
            policy_chosen_logps=output.policy_chosen_logps,
            policy_rejected_logps=output.policy_rejected_logps,
            reference_chosen_logps=output.reference_chosen_logps,
            reference_rejected_logps=output.reference_rejected_logps,
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
