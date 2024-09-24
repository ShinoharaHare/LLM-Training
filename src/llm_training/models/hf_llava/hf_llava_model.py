import torch
from torch import nn
from transformers import LlavaConfig, LlavaForConditionalGeneration

from llm_training.models.hf_compat_model import HFCompatModel
from llm_training.utils.decorators import copy_method_signature

from .hf_llava_config import HFLlavaConfig


class HFLlava(HFCompatModel):
    config: HFLlavaConfig
    hf_config: LlavaConfig
    hf_model: LlavaForConditionalGeneration

    config_class = HFLlavaConfig
    hf_config_class = LlavaConfig
    hf_model_class = LlavaForConditionalGeneration

    @property
    def no_split_modules(self) -> list[str] | None:
        return self.hf_model._no_split_modules

    def __init__(self, config: HFLlavaConfig) -> None:
        super().__init__(config)

        self.hf_model = self.construct_hf_model()

        if self.config.enable_gradient_checkpointing:
            self.hf_model.gradient_checkpointing_enable({'use_reentrant': False})

    def convert_state_dict_from_hf(self, hf_state_dict: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        return {'hf_model.' + k: v for k, v in hf_state_dict.items()}

    def convert_state_dict_to_hf(self, state_dict: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        return {k.removeprefix('hf_model.'): v for k, v in state_dict.items()}

    def forward(
        self,
        input_ids: torch.Tensor | None = None,
        pixel_values: torch.Tensor | None = None,
        attention_mask: torch.Tensor | None = None,
        position_ids: torch.Tensor | None = None,
        input_embeds: torch.Tensor | None = None,
        labels: torch.Tensor | None = None
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        vision_feature_layer = self.hf_config.vision_feature_layer
        vision_feature_select_strategy = self.hf_config.vision_feature_select_strategy

        if input_embeds is None:
            # 1. Extra the input embeddings
            input_embeds = self.get_input_embeddings()(input_ids)

            # 2. Merge text and images
            if pixel_values is not None and input_ids.shape[1] != 1:
                image_outputs = self.hf_model.vision_tower(pixel_values, output_hidden_states=True)
                # this is not memory efficient at all (output_hidden_states=True) will save all the hidden stated.
                selected_image_feature = image_outputs.hidden_states[vision_feature_layer]

                if vision_feature_select_strategy == "default":
                    selected_image_feature = selected_image_feature[:, 1:]
                elif vision_feature_select_strategy == "full":
                    selected_image_feature = selected_image_feature
                else:
                    raise ValueError(
                        f"Unexpected select feature strategy: {self.hf_config.vision_feature_select_strategy}"
                    )

                image_features = self.hf_model.multi_modal_projector(selected_image_feature)
                input_embeds = input_embeds.to(image_features.dtype)
                input_embeds, attention_mask, labels, position_ids = self.hf_model._merge_input_ids_with_image_features(
                    image_features, input_embeds, input_ids, attention_mask, labels
                )

        outputs = self.hf_model.language_model(
            attention_mask=attention_mask,
            position_ids=position_ids,
            inputs_embeds=input_embeds
        )

        logits = outputs[0]
        return attention_mask, labels, logits

    @copy_method_signature(forward)
    def __call__(): ...

    def get_input_embeddings(self) -> nn.Embedding:
        return self.hf_model.get_input_embeddings()
    
    def get_output_embeddings(self) -> nn.Linear:
        return self.hf_model.get_output_embeddings()
