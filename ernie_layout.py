# coding=utf-8
# Copyright 2018 The Google AI Language Team Authors and The HuggingFace Inc. team.
# Copyright (c) 2018, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""PyTorch ERNIE-Layout model."""


import math
import os
from typing import List, Optional, Tuple, Union

import torch
import torch.utils.checkpoint
from packaging import version
from torch import nn
from transformers import PretrainedConfig

from transformers.modeling_utils import (
    PreTrainedModel
)


class ErnieLayoutConfig(PretrainedConfig):
    r"""
    This is the configuration class to store the configuration of a [`ErnieLayoutModel`]. It is used to
    instantiate a ErnieLayout model according to the specified arguments, defining the model architecture. Instantiating a
    configuration with the defaults will yield a similar configuration to that of the ErnieLayout
    ernie-layoutx-base-uncased architecture.
    Configuration objects inherit from [`PretrainedConfig`] and can be used to control the model outputs. Read the
    documentation from [`PretrainedConfig`] for more information.
    Args:
        vocab_size (`int`, *optional*, defaults to 250002):
            Vocabulary size of the ErnieLayout model. Defines the number of different tokens that can be represented by the
            `inputs_ids` passed when calling [`ErnieLayoutModel`].
        hidden_size (`int`, *optional*, defaults to 768):
            Dimensionality of the encoder layers and the pooler layer.
        num_hidden_layers (`int`, *optional*, defaults to 12):
            Number of hidden layers in the Transformer encoder.
        num_attention_heads (`int`, *optional*, defaults to 12):
            Number of attention heads for each attention layer in the Transformer encoder.
        intermediate_size (`int`, *optional*, defaults to 3072):
            Dimensionality of the "intermediate" (often named feed-forward) layer in the Transformer encoder.
        hidden_act (`str` or `Callable`, *optional*, defaults to `"gelu"`):
            The non-linear activation function (function or string) in the encoder and pooler. If string, `"gelu"`,
            `"relu"`, `"silu"` and `"gelu_new"` are supported.
        hidden_dropout_prob (`float`, *optional*, defaults to 0.1):
            The dropout probability for all fully connected layers in the embeddings, encoder, and pooler.
        attention_probs_dropout_prob (`float`, *optional*, defaults to 0.1):
            The dropout ratio for the attention probabilities.
        max_position_embeddings (`int`, *optional*, defaults to 514):
            The maximum sequence length that this model might ever be used with. Typically set this to something large
            just in case (e.g., 514 or 1028 or 2056).
        type_vocab_size (`int`, *optional*, defaults to 100):
            The vocabulary size of the `token_type_ids` passed when calling [`ErnieModel`].
        initializer_range (`float`, *optional*, defaults to 0.02):
            The standard deviation of the truncated_normal_initializer for initializing all weight matrices.
        layer_norm_eps (`float`, *optional*, defaults to 1e-12):
            The epsilon used by the layer normalization layers.
        use_cache (`bool`, *optional*, defaults to `True`):
            Whether or not the model should return the last key/values attentions (not used by all models). Only
            relevant if `config.is_decoder=True`.
        classifier_dropout (`float`, *optional*):
            The dropout ratio for classifier.
        has_visual_segment_embedding (`bool`, *optional*, defaults to `False`):
            Whether or not the model has visual segment embedding.
    Examples:
    ```python
    >>> from paddlenlp.transformers import ErnieLayoutModel, ErnieLayoutConfig
    >>> # Initializing a ErnieLayout ernie-layoutx-base-uncased configuration
    >>> configuration = ErnieLayoutConfig()
    >>> # Initializing a model from the  style configuration
    >>> model = ErnieLayoutModel(configuration)
    >>> # Accessing the model configuration
    >>> configuration = model.config
    ```
    """
    model_type = "ernie_layout"

    def __init__(
        self,
        vocab_size=30522,
        hidden_size=768,
        num_hidden_layers=12,
        num_attention_heads=12,
        task_id=0,
        intermediate_size=3072,
        hidden_act="gelu",
        hidden_dropout_prob=0.1,
        attention_probs_dropout_prob=0.1,
        max_position_embeddings=512,
        max_2d_position_embeddings=1024,
        task_type_vocab_size=3,
        type_vocab_size=16,
        initializer_range=0.02,
        pad_token_id=0,
        pool_act = "tanh",
        fuse = False,
        image_feature_pool_shape = [7, 7, 256],
        # position_embedding_type="absolute",
        layer_norm_eps=1e-12,
        use_cache=False,
        use_task_id=True,
        classifier_dropout=None,
        has_visual_segment_embedding=False,
        position_embedding_type="absolute",
        **kwargs
    ):
        super().__init__(pad_token_id=pad_token_id, **kwargs)
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.task_id = task_id
        self.intermediate_size = intermediate_size
        self.hidden_act = hidden_act
        self.hidden_dropout_prob = hidden_dropout_prob
        self.attention_probs_dropout_prob = attention_probs_dropout_prob
        self.max_position_embeddings = max_position_embeddings
        self.max_2d_position_embeddings = max_2d_position_embeddings
        self.task_type_vocab_size = task_type_vocab_size
        self.type_vocab_size = type_vocab_size
        self.initializer_range = initializer_range
        self.pool_act = pool_act
        self.fuse = fuse
        self.image_feature_pool_shape = image_feature_pool_shape
        self.layer_norm_eps = layer_norm_eps
        self.use_cache = use_cache
        self.use_task_id = use_task_id
        self.classifier_dropout = classifier_dropout
        self.has_visual_segment_embedding = has_visual_segment_embedding
        self.position_embedding_type = position_embedding_type

logger = logging.get_logger(__name__)

_CHECKPOINT_FOR_DOC = "ernie-layout-base-uncased"
_CONFIG_FOR_DOC = "ErnieLayoutConfig"
_TOKENIZER_FOR_DOC = "ErnieLayoutTokenizer"

ERNIE_LAYOUT_PRETRAINED_MODEL_ARCHIVE_LIST = [
    # See all ERNIE models at https://huggingface.co/models?filter=ernie
]


class ErnieLayoutEmbeddings(nn.Module):
    """"""
    def __init__(self, config: ErnieLayoutConfig):
        super(ErnieLayoutEmbeddings, self).__init__()
        self.word_embeddings = nn.Embedding(config.vocab_size, config.hidden_size)
        self.position_embeddings = nn.Embedding(config.max_position_embeddings, config.hidden_size)

        self.x_position_embeddings = nn.Embedding(config.max_2d_position_embeddings, config.hidden_size)
        self.y_position_embeddings = nn.Embedding(config.max_2d_position_embeddings, config.hidden_size)
        self.h_position_embeddings = nn.Embedding(config.max_2d_position_embeddings, config.hidden_size)
        self.w_position_embeddings = nn.Embedding(config.max_2d_position_embeddings, config.hidden_size)

        self.token_type_embeddings = nn.Embedding(config.type_vocab_size, config.hidden_size)
        self.LayerNorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

        self.position_embedding_type = getattr(config, "position_embedding_type", "absolute")
        self.register_buffer("position_ids", torch.arange(config.max_position_embeddings).expand((1, -1)))

    def _cal_spatial_position_embeddings(self, bbox):
        try:
            left_position_embeddings = self.x_position_embeddings(bbox[:, :, 0])
            upper_position_embeddings = self.y_position_embeddings(bbox[:, :, 1])
            right_position_embeddings = self.x_position_embeddings(bbox[:, :, 2])
            lower_position_embeddings = self.y_position_embeddings(bbox[:, :, 3])
        except IndexError as e:
            raise IndexError("The :obj:`bbox` coordinate values should be within 0-1000 range.")
        h_position_embeddings = self.h_position_embeddings(bbox[:, :, 3] - bbox[:, :, 1])
        w_position_embeddings = self.w_position_embeddings(bbox[:, :, 2] - bbox[:, :, 0])
        return (
            left_position_embeddings,
            upper_position_embeddings,
            right_position_embeddings,
            lower_position_embeddings,
            h_position_embeddings,
            w_position_embeddings,
        )

    def forward(self, input_ids, bbox=None, token_type_ids=None, position_ids=None):
        if position_ids is None:
            ones = torch.ones_like(input_ids, dtype=torch.int64)
            seq_length = torch.cumsum(ones, dim=-1)
            position_ids = seq_length - ones
            # position_ids.stop_gradient = True
        if token_type_ids is None:
            token_type_ids = torch.zeros_like(input_ids, dtype=torch.int64)

        input_embeddings = self.word_embeddings(input_ids)
        position_embeddings = self.position_embeddings(position_ids, require_grad=True)

        x1, y1, x2, y2, h, w = self.embeddings._cal_spatial_position_embeddings(bbox)
        token_type_embeddings = self.token_type_embeddings(token_type_ids)

        embeddings = input_embeddings + position_embeddings + x1 + y1 + x2 + y2 + h + w + token_type_embeddings

        embeddings = self.LayerNorm(embeddings)
        embeddings = self.dropout(embeddings)
        return embeddings

class ErnieLayoutSelfOutput(nn.Module):
    def __init__(self, config: ErnieLayoutConfig):
        super(ErnieLayoutSelfOutput, self).__init__()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.LayerNorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    def forward(self, hidden_states, input_tensor):
        hidden_states = self.dense(hidden_states)
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.LayerNorm(hidden_states + input_tensor)
        return hidden_states


class ErnieLayoutSelfAttention(nn.Module):
    def __init__(self, config: ErnieLayoutConfig):
        super(ErnieLayoutSelfAttention, self).__init__()
        if config.hidden_size % config.num_attention_heads != 0 and not hasattr(config, "embedding_size"):
            raise ValueError(
                f"The hidden size ({config.hidden_size}) is not a multiple of the number of attention "
                f"heads ({config.num_attention_heads})"
            )
        self.num_attention_heads = config.num_attention_heads
        self.attention_head_size = int(config.hidden_size / config.num_attention_heads)
        self.all_head_size = self.num_attention_heads * self.attention_head_size

        self.has_relative_attention_bias = config.has_relative_attention_bias
        self.has_spatial_attention_bias = config.has_spatial_attention_bias

        self.query = nn.Linear(config.hidden_size, self.all_head_size)
        self.key = nn.Linear(config.hidden_size, self.all_head_size)
        self.value = nn.Linear(config.hidden_size, self.all_head_size)

        self.dropout = nn.Dropout(config.attention_probs_dropout_prob)

    def transpose_for_scores(self, x):
        x = x.reshape([x.shape[0], x.shape[1], self.num_attention_heads, self.attention_head_size])
        return x.transpose([0, 2, 1, 3])

    def compute_qkv(self, hidden_states):
        q = self.query(hidden_states)
        k = self.key(hidden_states)
        v = self.value(hidden_states)
        return q, k, v

    def forward(
        self,
        hidden_states,
        attention_mask=None,
        head_mask=None,
        encoder_hidden_states=None,
        encoder_attention_mask=None,
        past_key_value=None,
        output_attentions=False,
        rel_pos=None,
        rel_2d_pos=None
    ):
        q, k, v = self.compute_qkv(hidden_states)

        # (B, L, H*D) -> (B, H, L, D)
        query_layer = self.transpose_for_scores(q)
        key_layer = self.transpose_for_scores(k)
        value_layer = self.transpose_for_scores(v)

        query_layer = query_layer / math.sqrt(self.attention_head_size)
        # [BSZ, NAT, L, L]
        attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))

        if self.has_relative_attention_bias:
            attention_scores += rel_pos
        if self.has_spatial_attention_bias:
            attention_scores += rel_2d_pos
        bool_attention_mask = attention_mask.astype(torch.bool)
        bool_attention_mask.stop_gradient = True
        attention_scores_shape = attention_scores.shape
        attention_scores = torch.where(
            bool_attention_mask.expand_as(attention_scores_shape),
            torch.ones(attention_scores_shape).type_as(attention_scores) * float('-1e10'),
            attention_scores,
        )
        attention_probs = F.softmax(attention_scores, dim=-1)
        # This is actually dropping out entire tokens to attend to, which might
        # seem a bit unusual, but is taken from the original Transformer paper.
        attention_probs = self.dropout(attention_probs)

        context_layer = torch.matmul(attention_probs, value_layer)
        context_layer = context_layer.transpose([0, 2, 1, 3])
        new_context_layer_shape = context_layer.shape
        context_layer = context_layer.reshape([new_context_layer_shape[0], new_context_layer_shape[1], self.all_head_size])
        outputs = (context_layer, attention_probs) if output_attentions else (context_layer,)
        return outputs


class ErnieLayoutAttention(nn.Module):
    def __init__(self, config: ErnieLayoutConfig):
        super(ErnieLayoutAttention, self).__init__()
        self.self = ErnieLayoutSelfAttention(config)
        self.output = ErnieLayoutSelfOutput(config)

    def forward(
        self,
        hidden_states,
        attention_mask=None,
        head_mask=None,
        encoder_hidden_states=None,
        encoder_attention_mask=None,
        past_key_value=None,
        output_attentions=False,
        rel_pos=None,
        rel_2d_pos=None
    ):
        self.outputs = self.self(
            hidden_states,
            attention_mask,
            head_mask,
            encoder_hidden_states,
            encoder_attention_mask,
            past_key_value,
            output_attentions,
            rel_pos=rel_pos,
            rel_2d_pos=rel_2d_pos,
        )
        attention_output = self.output(self.outputs[0], hidden_states)
        # add attentions if we output them
        if attention_output:
            outputs = (attention_output,) + self.outputs[1:]
        else:
            outputs = (attention_output,)
        return outputs


class ErnieLayoutEncoder(nn.Module):
    def __init__(self, config: ErnieLayoutConfig):
        super(ErnieLayoutEncoder, self).__init__()
        self.config = config
        # Recompute defaults to False and is controlled by Trainer
        self.enable_recompute = False
        self.layer = nn.ModuleList([ErnieLayoutAttention(config) for _ in range(config.num_hidden_layers)])

        self.has_relative_attention_bias = config.has_relative_attention_bias
        self.has_spatial_attention_bias = config.has_spatial_attention_bias
        if self.has_relative_attention_bias:
            self.rel_pos_bins = config.rel_pos_bins
            self.max_rel_pos = config.max_rel_pos
            self.rel_pos_onehot_size = config.rel_pos_bins
            self.rel_pos_bias = nn.parameter.Parameter(
                torch.zeros(self.rel_pos_onehot_size, self.max_rel_pos),
                dtype=torch.float32,
                requires_grad=True
            )
        if self.has_spatial_attention_bias:
            self.max_rel_2d_pos = config.max_rel_2d_pos
            self.rel_2d_pos_bins = config.rel_2d_pos_bins
            self.rel_2d_pos_onehot_size = config.rel_2d_pos_bins
            self.rel_pos_x_bias = nn.parameter.Parameter(
                torch.zeros(self.rel_2d_pos_onehot_size, self.num_attention_heads),
                dtype=torch.float32,
                requires_grad=True
            )
            self.rel_pos_y_bias = nn.parameter.Parameter(
                torch.zeros(self.rel_2d_pos_onehot_size, self.num_attention_heads),
                dtype=torch.float32,
                requires_grad=True
            )

    def _cal_1d_pos_emb(self, hidden_states, position_ids):
        rel_pos_mat = position_ids.unsqueeze(-2) - position_ids.unsqueeze(-1)
        rel_pos = relative_position_bucket(
            rel_pos_mat,
            num_buckets=self.rel_pos_bins,
            max_distance=self.max_rel_pos
        )
        rel_pos = F.one_hot(rel_pos, num_classes=self.rel_pos_onehot_size).type_as(hidden_states)
        # rel_pos = rel_pos.to(torch.float32)
        rel_pos = torch.matmul(rel_pos, self.rel_pos_bias)
        rel_pos.transpose([0, 3, 1, 2])
        return rel_pos

    def _cal_2d_pos_emb(self, hidden_states, bbox):
        position_coord_x = bbox[:, :, 0]
        position_coord_y = bbox[:, :, 3]
        rel_pos_x_2d_max = position_coord_x.unsqueeze(-2) - position_coord_x.unsqueeze(-1)
        rel_pos_y_2d_max = position_coord_y.unsqueeze(-2) - position_coord_y.unsqueeze(-1)
        rel_pos_x = relative_position_bucket(
            rel_pos_x_2d_max,
            num_buckets=self.rel_2d_pos_bins,
            max_distance=self.max_rel_2d_pos
        )
        rel_pos_y = relative_position_bucket(
            rel_pos_y_2d_max,
            num_buckets=self.rel_2d_pos_bins,
            max_distance=self.max_rel_2d_pos
        )
        rel_pos_x = F.one_hot(rel_pos_x, num_classes=self.rel_2d_pos_onehot_size).type_as(hidden_states)
        rel_pos_y = F.one_hot(rel_pos_y, num_classes=self.rel_2d_pos_onehot_size).type_as(hidden_states)
        rel_pos_x = torch.matmul(rel_pos_x, self.rel_pos_x_bias)
        rel_pos_y = torch.matmul(rel_pos_y, self.rel_pos_y_bias)
        rel_pos = rel_pos_x + rel_pos_y
        return rel_pos

    def forward(
        self,
        hidden_states,
        attention_mask=None,
        head_mask=None,
        encoder_hidden_states=None,
        encoder_attention_mask=None,
        past_key_values=None,
        output_attentions=False,
        output_hidden_states=False,
        bbox=None,
        position_ids=None,
    ):
        all_hidden_states = () if output_hidden_states else None
        rel_pos = self._cal_1d_pos_emb(hidden_states, position_ids) if self.has_relative_attention_bias else None
        rel_2d_pos = self._cal_2d_pos_emb(hidden_states, bbox) if self.has_spatial_attention_bias else None

        hidden_save = dict()
        hidden_save["input_hidden_states"] = hidden_states

        for i, layer_module in enumerate(self.layer):
            if output_hidden_states:
                all_hidden_states = all_hidden_states + (hidden_states,)
            layer_head_mask = head_mask[i] if head_mask is not None else None
            past_key_value = past_key_values[i] if past_key_values is not None else None

            #
            hidden_save["input_hidden_mask"] = hidden_mask
            hidden_save["input_hidden_head_mask"] = layer_head_mask

            if self.enable_recompute and self.training:
                def create_custom_forward(module):
                    def custom_forward(*inputs):
                        return module(*inputs, past_key_value, output_attentions)

                    return custom_forward
                layer_outputs = torch.utils.checkpoint.checkpoint(
                    create_custom_forward(layer_module),
                    hidden_states,
                    attention_mask,
                    layer_head_mask,
                    encoder_hidden_states,
                    encoder_attention_mask,
                    rel_pos,
                    rel_2d_pos
                )
            else:
                layer_outputs = layer_module(
                    hidden_states,
                    attention_mask,
                    layer_head_mask,
                    encoder_hidden_states,
                    encoder_attention_mask,
                    past_key_value,
                    output_attentions,
                    rel_pos,
                    rel_2d_pos
                )
            hidden_states = layer_outputs[0]
            if use_cache:
                next_decoder_cache += (layer_outputs[-1],)
            if output_attentions:
                all_self_attentions = all_self_attentions + (layer_outputs[1],)
                if self.config.add_cross_attention:
                    all_cross_attentions = all_cross_attentions + \
                        (layer_outputs[2],)
        return (hidden_states,)


class ErnieLayoutIntermediate(nn.Module):
    pass


class ErnieLayoutOutput(nn.Module):
    pass


class ErnieLayoutLayer(nn.Module):
    pass


class VisualBackbone(nn.Module):
    pass


class ErnieLayoutPretrainedModel(PretrainedModel):
    model_config_file = CONFIG_NAME
    pass


class ErnieLayoutModel(ErnieLayoutPretrainedModel):
    pass
