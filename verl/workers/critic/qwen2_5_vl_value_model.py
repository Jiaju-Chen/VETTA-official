"""Token-aligned value heads on a Qwen2.5-VL visual-language backbone."""

import torch
from torch import nn
from transformers.modeling_outputs import TokenClassifierOutput
from transformers.models.qwen2_5_vl.modeling_qwen2_5_vl import (
    Qwen2_5_VLModel,
    Qwen2_5_VLPreTrainedModel,
    Qwen2_5_VisionTransformerPretrainedModel,
)


class Qwen2_5_VLForTokenValues(Qwen2_5_VLPreTrainedModel):
    _no_split_modules = ["Qwen2_5_VLDecoderLayer", "Qwen2_5_VLVisionBlock"]

    def __init__(self, config):
        super().__init__(config)
        self.visual = Qwen2_5_VisionTransformerPretrainedModel._from_config(config.vision_config)
        self.model = Qwen2_5_VLModel(config)
        self.score = nn.Linear(config.hidden_size, config.num_labels, bias=False)
        self.post_init()

    def get_input_embeddings(self):
        return self.model.embed_tokens

    def set_input_embeddings(self, value):
        self.model.embed_tokens = value

    def forward(
        self,
        input_ids,
        attention_mask=None,
        position_ids=None,
        pixel_values=None,
        image_grid_thw=None,
        use_cache=False,
        **kwargs,
    ):
        if position_ids is None:
            raise ValueError("Qwen2.5-VL Critic requires multimodal position_ids")
        if pixel_values is not None and image_grid_thw is None:
            raise ValueError("image_grid_thw is required with pixel_values")
        if kwargs.get("pixel_values_videos") is not None:
            raise ValueError("Video input is not supported by this Critic")
        if position_ids.ndim == 3 and position_ids.shape[0] == 4:
            position_ids = position_ids[1:]
        if position_ids.ndim != 3 or position_ids.shape[0] != 3:
            raise ValueError("Qwen2.5-VL Critic expects 3 visual position channels")

        inputs_embeds = self.model.embed_tokens(input_ids)
        if pixel_values is not None:
            image_embeds = self.visual(pixel_values.to(self.visual.dtype), grid_thw=image_grid_thw)
            vision_start = getattr(self.config, "vision_start_token_id", None)
            vision_end = getattr(self.config, "vision_end_token_id", None)
            if vision_start is None or vision_end is None:
                raise ValueError("Qwen2.5-VL Critic requires vision span token ids")
            inside_vision_span = (input_ids == vision_start).long().cumsum(dim=-1) > (
                input_ids == vision_end
            ).long().cumsum(dim=-1)
            image_mask = (input_ids == self.config.image_token_id) & inside_vision_span
            if image_mask.sum().item() != image_embeds.shape[0]:
                raise ValueError(
                    "Image feature and in-span image-token counts differ: "
                    f"{image_embeds.shape[0]} features for {image_mask.sum().item()} tokens"
                )
            inputs_embeds = inputs_embeds.masked_scatter(
                image_mask.unsqueeze(-1).expand_as(inputs_embeds),
                image_embeds.to(device=inputs_embeds.device, dtype=inputs_embeds.dtype),
            )

        hidden_states = self.model(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            position_ids=position_ids,
            use_cache=use_cache,
            return_dict=True,
        ).last_hidden_state
        return TokenClassifierOutput(logits=self.score(hidden_states))
