from types import SimpleNamespace

import torch
from torch import nn

from verl.models.transformers.qwen2_vl import _get_input_embeds


class FakeVisual:
    dtype = torch.float32

    def __call__(self, pixel_values, grid_thw):
        return torch.full((162, 4), 7.0, dtype=self.dtype)


class FakeModel:
    def __init__(self):
        self.config = SimpleNamespace(
            image_token_id=11,
            vision_start_token_id=10,
            vision_end_token_id=12,
        )
        self.visual = FakeVisual()
        self.embedding = nn.Embedding(16, 4)
        with torch.no_grad():
            self.embedding.weight.fill_(0.25)

    def get_input_embeddings(self):
        return self.embedding


def test_generated_image_pad_outside_vision_span_is_not_used_as_image_feature():
    model = FakeModel()
    input_ids = torch.tensor([[10] + [11] * 162 + [12, 11]])

    embeddings, _ = _get_input_embeds(
        model,
        input_ids,
        attention_mask=torch.ones_like(input_ids),
        pixel_values=torch.zeros((1, 4)),
        image_grid_thw=torch.tensor([[1, 18, 18]]),
    )

    torch.testing.assert_close(embeddings[0, 1:163], torch.full((162, 4), 7.0))
    torch.testing.assert_close(embeddings[0, -1], torch.full((4,), 0.25))
