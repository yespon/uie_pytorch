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

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import CrossEntropyLoss
from transformers.modeling_utils import PretrainedConfig, PreTrainedModel
from transformers.utils import logging

from visual_backbone import ResNetCustomized

logger = logging.get_logger(__name__)

__all__ = [
    "ErnieLayoutModel",
    "ErnieLayoutPretrainedModel",
]


class ErnieLayoutConfig(PretrainedConfig):
    model_type = "ernie_layout"

    def __init__(
            self,
            vocab_size=25002,
            hidden_size=768,
            num_hidden_layers=12,
            num_attention_heads=12,
            intermediate_size=3072,
            hidden_act="gelu",
            hidden_dropout_prob=0.1,
            attention_probs_dropout_prob=0.1,
            max_position_embeddings=514,
            type_vocab_size=100,
            initializer_range=0.02,
            layer_norm_eps=1e-12,
            pad_token_id=1,
            bos_token_id=0,
            eos_token_id=2,
            max_2d_position_embeddings=1024,
            coordinate_size=128,
            shape_size=128,
            has_relative_attention_bias=True,
            rel_pos_bins=32,
            max_rel_pos=128,
            rel_2d_pos_bins=64,
            max_rel_2d_pos=256,
            has_spatial_attention_bias=True,
            input_size=224,
            classifier_dropout=None,
            gradient_checkpointing=False,
            has_visual_segment_embedding=False,
            image_feature_pool_shape=[7, 7, 256],
            output_past=True,
            **kwargs
    ):
        super().__init__(
            vocab_size=vocab_size,
            hidden_size=hidden_size,
            num_hidden_layers=num_hidden_layers,
            num_attention_heads=num_attention_heads,
            intermediate_size=intermediate_size,
            hidden_act=hidden_act,
            hidden_dropout_prob=hidden_dropout_prob,
            attention_probs_dropout_prob=attention_probs_dropout_prob,
            max_position_embeddings=max_position_embeddings,
            type_vocab_size=type_vocab_size,
            initializer_range=initializer_range,
            layer_norm_eps=layer_norm_eps,
            pad_token_id=pad_token_id,
            bos_token_id=bos_token_id,
            eos_token_id=eos_token_id,
            **kwargs,
        )
        self.classifier_dropout = classifier_dropout
        self.coordinate_size = coordinate_size
        self.gradient_checkpointing = gradient_checkpointing
        self.has_relative_attention_bias = has_relative_attention_bias
        self.has_spatial_attention_bias = has_spatial_attention_bias
        self.has_visual_segment_embedding = has_visual_segment_embedding
        self.input_size = input_size
        self.image_feature_pool_shape = image_feature_pool_shape
        self.max_rel_pos = max_rel_pos
        self.max_rel_2d_pos = max_rel_2d_pos
        self.max_2d_position_embeddings = max_2d_position_embeddings
        self.output_past = output_past
        self.rel_pos_bins = rel_pos_bins
        self.rel_2d_pos_bins = rel_2d_pos_bins
        self.shape_size = shape_size


def relative_position_bucket(relative_position,
                             bidirectional=True,
                             num_buckets=32,
                             max_distance=128):
    """
    Adapted from Mesh Tensorflow:
    https://github.com/tensorflow/mesh/blob/0cb87fe07da627bf0b7e60475d59f95ed6b5be3d/mesh_tensorflow/transformer/transformer_layers.py#L593
    Translate relative position to a bucket number for relative attention. The relative position is defined as
    memory_position - query_position, i.e. the distance in tokens from the attending position to the attended-to
    position. If bidirectional=False, then positive relative positions are invalid. We use smaller buckets for small
    absolute relative_position and larger buckets for larger absolute relative_positions. All relative positions
    >=max_distance map to the same bucket. All relative positions <=-max_distance map to the same bucket. This should
    allow for more graceful generalization to longer sequences than the model has been trained on.

    Args:
        relative_position: an int32 Tensor
        bidirectional: a boolean - whether the attention is bidirectional
        num_buckets: an integer
        max_distance: an integer

    Returns:
        a Tensor with the same shape as relative_position, containing int32 values in the range [0, num_buckets)
    """

    ret = 0
    if bidirectional:
        num_buckets //= 2
        ret += (relative_position > 0).long() * num_buckets
        n = torch.abs(relative_position)
    else:
        n = torch.max(-relative_position, torch.zeros_like(relative_position))
    # Now n is in the range [0, inf)
    # half of the buckets are for exact increments in positions
    max_exact = num_buckets // 2
    is_small = n < max_exact

    # The other half of the buckets are for logarithmically bigger bins in positions up to max_distance
    val_if_large = max_exact + (
            torch.log(n.float() / max_exact) / math.log(max_distance / max_exact) * (num_buckets - max_exact)
    ).to(torch.long)

    val_if_large = torch.minimum(
        val_if_large, torch.full_like(val_if_large, num_buckets - 1))

    ret += torch.where(is_small, n, val_if_large)
    return ret


class ErnieLayoutPooler(nn.Module):

    def __init__(self, hidden_size, with_pool):
        super(ErnieLayoutPooler, self).__init__()
        self.dense = nn.Linear(hidden_size, hidden_size)
        self.activation = nn.Tanh()
        self.with_pool = with_pool

    def forward(self, hidden_states):
        # We "pool" the model by simply taking the hidden state corresponding
        # to the first token.
        first_token_tensor = hidden_states[:, 0]
        pooled_output = self.dense(first_token_tensor)
        if self.with_pool == 'tanh':
            pooled_output = self.activation(pooled_output)
        return pooled_output


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

        self.register_buffer("position_ids", torch.arange(config.max_position_embeddings).expand((1, -1)))

    def _cal_spatial_position_embeddings(self, bbox):
        try:
            left_position_embeddings = self.x_position_embeddings(bbox[:, :, 0])
            upper_position_embeddings = self.y_position_embeddings(bbox[:, :, 1])
            right_position_embeddings = self.x_position_embeddings(bbox[:, :, 2])
            lower_position_embeddings = self.y_position_embeddings(bbox[:, :, 3])
        except IndexError as e:
            raise IndexError("The :obj:`bbox` coordinate values should be within 0-1000 range.") from e
        h_position_embeddings = self.h_position_embeddings(bbox[:, :, 3] - bbox[:, :, 1])
        w_position_embeddings = self.w_position_embeddings(bbox[:, :, 2] - bbox[:, :, 0])
        return left_position_embeddings, upper_position_embeddings, right_position_embeddings, \
            lower_position_embeddings, h_position_embeddings, w_position_embeddings

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

        self.dropout_p = config.attention_probs_dropout_prob
        self.use_flash_attn = getattr(config, "use_flash_attn", False)
        if not self.use_flash_attn:
            self.dropout = nn.Dropout(self.dropout_p)

    def transpose_for_scores(self, x):
        new_x_shape = x.size()[:-1] + (
            self.num_attention_heads,
            self.attention_head_size,
        )
        x = x.view(*new_x_shape)
        return x.permute(0, 2, 1, 3)

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
            output_attentions=False,
            rel_pos=None,
            rel_2d_pos=None
    ):
        q, k, v = self.compute_qkv(hidden_states)

        # (B, L, H*D) -> (B, H, L, D)
        query_layer = self.transpose_for_scores(q)
        key_layer = self.transpose_for_scores(k)
        value_layer = self.transpose_for_scores(v)

        if self.use_flash_attn:
            logger.info("use flash attention")
            attention_probs = None
            bz, _, seq_len, _ = key_layer.size()
            attn_bias = torch.zeros((bz, 1, seq_len, seq_len), dtype=key_layer.dtype, device=key_layer.device)
            attn_bias.masked_fill_(attention_mask, float("-inf"))
            if self.has_relative_attention_bias:
                attn_bias = attn_bias + rel_pos
            if self.has_spatial_attention_bias:
                attn_bias = attn_bias + rel_2d_pos
            with torch.backends.cuda.sdp_kernel(enable_flash=True, enable_mem_efficient=True):
                context_layer = F.scaled_dot_product_attention(query_layer, key_layer, value_layer,
                                                               dropout_p=self.dropout_p if self.training else 0.0,
                                                               attn_mask=attn_bias)
        else:
            query_layer = query_layer / math.sqrt(self.attention_head_size)
            # [BSZ, NAT, L, L]
            attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))

            if self.has_relative_attention_bias:
                attention_scores += rel_pos
            if self.has_spatial_attention_bias:
                attention_scores += rel_2d_pos
            attention_scores = attention_scores.float().masked_fill_(
                attention_mask.to(torch.bool), torch.finfo(attention_scores.dtype).min
            )
            attention_probs = nn.functional.softmax(attention_scores, dim=-1, dtype=torch.float32).type_as(value_layer)
            # This is actually dropping out entire tokens to attend to, which might
            # seem a bit unusual, but is taken from the original Transformer paper.
            attention_probs = self.dropout(attention_probs)
            # Mask heads if we want to
            if head_mask is not None:
                attention_probs = attention_probs * head_mask
            context_layer = torch.matmul(attention_probs, value_layer)

        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        new_context_layer_shape = context_layer.size()[:-2] + (self.all_head_size,)
        context_layer = context_layer.view(*new_context_layer_shape)
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
            output_attentions=False,
            rel_pos=None,
            rel_2d_pos=None
    ):
        self_outputs = self.self(
            hidden_states,
            attention_mask,
            head_mask,
            output_attentions,
            rel_pos=rel_pos,
            rel_2d_pos=rel_2d_pos,
        )
        attention_output = self.output(self_outputs[0], hidden_states)
        # add attentions if we output them
        if output_attentions:
            outputs = [attention_output, ] + self.outputs[1:]
        else:
            outputs = [attention_output]
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
            self.rel_pos_bias = nn.Linear(self.rel_pos_bins, config.num_attention_heads, bias=False)

        if self.has_spatial_attention_bias:
            self.max_rel_2d_pos = config.max_rel_2d_pos
            self.rel_2d_pos_bins = config.rel_2d_pos_bins
            self.rel_pos_x_bias = nn.Linear(self.rel_2d_pos_bins, config.num_attention_heads, bias=False)
            self.rel_pos_y_bias = nn.Linear(self.rel_2d_pos_bins, config.num_attention_heads, bias=False)

    def _cal_1d_pos_emb(self, position_ids):
        rel_pos_mat = position_ids.unsqueeze(-2) - position_ids.unsqueeze(-1)
        rel_pos = relative_position_bucket(
            rel_pos_mat,
            num_buckets=self.rel_pos_bins,
            max_distance=self.max_rel_pos
        )
        rel_pos = self.rel_pos_bias.weight.t()[rel_pos].permute(0, 3, 1, 2)
        rel_pos = rel_pos.contiguous()
        return rel_pos

    def _cal_2d_pos_emb(self, bbox):
        position_coord_x = bbox[:, :, 0]
        position_coord_y = bbox[:, :, 3]
        rel_pos_x_2d_mat = position_coord_x.unsqueeze(-2) - position_coord_x.unsqueeze(-1)
        rel_pos_y_2d_mat = position_coord_y.unsqueeze(-2) - position_coord_y.unsqueeze(-1)
        rel_pos_x = relative_position_bucket(
            rel_pos_x_2d_mat,
            num_buckets=self.rel_2d_pos_bins,
            max_distance=self.max_rel_2d_pos
        )
        rel_pos_y = relative_position_bucket(
            rel_pos_y_2d_mat,
            num_buckets=self.rel_2d_pos_bins,
            max_distance=self.max_rel_2d_pos
        )
        rel_pos_x = self.rel_pos_x_bias.weight.t()[rel_pos_x].permute(0, 3, 1, 2)
        rel_pos_y = self.rel_pos_y_bias.weight.t()[rel_pos_y].permute(0, 3, 1, 2)
        rel_pos_x = rel_pos_x.contiguous()
        rel_pos_y = rel_pos_y.contiguous()
        rel_2d_pos = rel_pos_x + rel_pos_y
        return rel_2d_pos

    def forward(
            self,
            hidden_states,
            attention_mask=None,
            head_mask=None,
            output_attentions=False,
            output_hidden_states=False,
            bbox=None,
            position_ids=None,
    ):
        all_hidden_states = () if output_hidden_states else None
        rel_pos = self._cal_1d_pos_emb(position_ids) if self.has_relative_attention_bias else None
        rel_2d_pos = self._cal_2d_pos_emb(bbox) if self.has_spatial_attention_bias else None

        hidden_save = dict()
        hidden_save["input_hidden_states"] = hidden_states

        for i, layer_module in enumerate(self.layer):
            if output_hidden_states:
                all_hidden_states = all_hidden_states + (hidden_states,)
            layer_head_mask = head_mask[i] if head_mask is not None else None

            #
            hidden_save["input_hidden_mask"] = attention_mask
            hidden_save["input_hidden_head_mask"] = layer_head_mask
            layer_outputs = layer_module(
                hidden_states,
                attention_mask,
                layer_head_mask,
                output_attentions,
                rel_pos=rel_pos,
                rel_2d_pos=rel_2d_pos,
            )

            hidden_states = layer_outputs[0]

            hidden_save["output_hidden_states"] = hidden_states
        return hidden_states,


class ErnieLayoutIntermediate(nn.Module):
    def __init__(self, config):
        super(ErnieLayoutIntermediate, self).__init__()
        self.dense = nn.Linear(config.hidden_size, config.intermediate_size)
        if config.hidden_act == "gelu":
            self.intermediate_act_fn = nn.GELU()
        else:
            assert False, "hidden_act is set as: {}, please check it..".format(
                config.hidden_act)

    def forward(self, hidden_states):
        hidden_states = self.dense(hidden_states)
        hidden_states = self.intermediate_act_fn(hidden_states)
        return hidden_states


class ErnieLayoutOutput(nn.Module):
    def __init__(self, config):
        super(ErnieLayoutOutput, self).__init__()
        self.dense = nn.Linear(config.intermediate_size, config.hidden_size)
        self.LayerNorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    def forward(self, hidden_states, input_tensor):
        hidden_states = self.dense(hidden_states)
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.LayerNorm(hidden_states + input_tensor)
        return hidden_states


class ErnieLayoutLayer(nn.Module):
    def __init__(self, config):
        super(ErnieLayoutLayer, self).__init__()
        # since chunk_size_feed_forward is 0 as default, no chunk is needed here.
        self.seq_len_dim = 1
        self.attention = ErnieLayoutAttention(config)
        self.add_cross_attention = False  # default as false
        self.intermediate = ErnieLayoutIntermediate(config)
        self.output = ErnieLayoutOutput(config)

    def feed_forward_chunk(self, attention_output):
        intermediate_output = self.intermediate(attention_output)
        layer_output = self.output(intermediate_output, attention_output)
        return layer_output

    def forward(
            self,
            hidden_states,
            attention_mask=None,
            head_mask=None,
            output_attentions=False,
            rel_pos=None,
            rel_2d_pos=None,
    ):
        self_attention_outputs = self.attention(
            hidden_states,
            attention_mask,
            head_mask,
            output_attentions=output_attentions,
            rel_pos=rel_pos,
            rel_2d_pos=rel_2d_pos,
        )
        attention_output = self_attention_outputs[0]
        layer_output = self.feed_forward_chunk(attention_output)

        if output_attentions:
            outputs = self_attention_outputs[1:]  # add self attentions if we output attention weights
            outputs = [layer_output, ] + list(outputs)
        else:
            outputs = [layer_output]
        return outputs


class VisualBackbone(nn.Module):

    def __init__(self, config: ErnieLayoutConfig):
        super(VisualBackbone, self).__init__()
        self.backbone = ResNetCustomized(layers=101)

        self.register_buffer(
            "pixel_mean",
            torch.tensor([103.53, 116.28, 123.675]).reshape([3, 1, 1]))
        self.register_buffer(
            "pixel_std",
            torch.tensor([57.375, 57.12, 58.395]).reshape([3, 1, 1]))

        self.pool = nn.AdaptiveAvgPool2d(config.image_feature_pool_shape[:2])

    def forward(self, images):
        images_input = (images - self.pixel_mean) / self.pixel_std
        features = self.backbone(images_input)
        features = self.pool(features).flatten(start_dim=2).transpose(1, 2).contiguous()
        return features


class ErnieLayoutPretrainedModel(PreTrainedModel):
    """
    An abstract class to handle weights initialization and a simple interface for downloading and loading pretrained
    models.
    """

    config_class = ErnieLayoutConfig
    base_model_prefix = "ernie_layout"
    supports_gradient_checkpointing = True
    _keys_to_ignore_on_load_missing = [r"position_ids"]

    def _init_weights(self, module):
        """Initialize the weights"""
        if isinstance(module, nn.Linear):
            # Slightly different from the TF version which uses truncated_normal for initialization
            # cf https://github.com/pytorch/pytorch/pull/5617
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.Embedding):
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)

    def _set_gradient_checkpointing(self, module, value=False):
        if isinstance(module, ErnieLayoutEncoder):
            module.gradient_checkpointing = value


class ErnieLayoutModel(ErnieLayoutPretrainedModel):
    """
        The bare ErnieLayout Model outputting raw hidden-states.

        This model inherits from :class:`~torchnlp.transformers.model_utils.PretrainedModel`.
        Refer to the superclass documentation for the generic methods.

        This model is also a Paddle `torch.nn.Layer <https://www.torchtorch.org.cn/documentation
        /docs/en/api/torch/fluid/dygraph/layers/Layer_en.html>`__ subclass. Use it as a regular Paddle Layer
        and refer to the Paddle documentation for all matter related to general usage and behavior.

        Args:
            vocab_size (`int`):
                Vocabulary size of the XLNet model. Defines the number of different tokens that can
                be represented by the `inputs_ids` passed when calling ErnieLayoutModel.
            hidden_size (`int`, optional):
                Dimensionality of the encoder layers and the pooler layer. Defaults to ``768``.
            num_hidden_layers (`int`, optional):
                Number of hidden layers in the Transformer encoder. Defaults to ``12``.
            num_attention_heads (`int`, optional):
                Number of attention heads for each attention layer in the Transformer encoder.
                Defaults to ``12``.
            intermediate_size (`int`, optional):
                Dimensionality of the "intermediate" (often named feed-forward) layer in the Transformer encoder.
                Defaults to ``3072``.
            hidden_act (`str`, optional):
                The non-linear activation function in the feed-forward layer.
                ``"gelu"``, ``"relu"`` and any other torch supported activation functions
                are supported. Defaults to ``"gelu"``.
            hidden_dropout_prob (`float`, optional):
                The dropout probability for all fully connected layers in the embeddings and encoder.
                Defaults to ``0.1``.
            attention_probs_dropout_prob (`float`, optional):
                The dropout probability for all fully connected layers in the pooler.
                Defaults to ``0.1``.
            initializer_range (`float`, optional):
                The standard deviation of the truncated_normal_initializer for initializing all weight matrices.
                Defaults to ``0.02``.
    """

    def __init__(self, config: ErnieLayoutConfig):
        super(ErnieLayoutModel, self).__init__(config)
        with_pool = 'tanh'
        self.config = config
        self.has_visual_segment_embedding = self.config.has_visual_segment_embedding
        self.embeddings = ErnieLayoutEmbeddings(config)

        self.visual = VisualBackbone(config)
        self.visual_proj = nn.Linear(config.image_feature_pool_shape[-1], config.hidden_size)
        self.visual_act_fn = nn.GELU()
        if self.has_visual_segment_embedding:
            self.visual_segment_embedding = nn.Parameter(nn.Embedding(1, config.hidden_size).weight[0])
        self.visual_LayerNorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.visual_dropout = nn.Dropout(config.hidden_dropout_prob)
        self.encoder = ErnieLayoutEncoder(config)
        self.pooler = ErnieLayoutPooler(config.hidden_size, with_pool)

        self.use_flash_attn = getattr(config, "use_flash_attn", False)

    def _calc_text_embeddings(self, input_ids, bbox, position_ids,
                              token_type_ids):
        words_embeddings = self.embeddings.word_embeddings(input_ids)
        position_embeddings = self.embeddings.position_embeddings(position_ids)
        x1, y1, x2, y2, h, w = self.embeddings._calc_spatial_position_embeddings(bbox)
        token_type_embeddings = self.embeddings.token_type_embeddings(token_type_ids)
        embeddings = words_embeddings + position_embeddings + x1 + y1 + x2 + y2 + w + h + token_type_embeddings

        embeddings = self.embeddings.LayerNorm(embeddings)
        embeddings = self.embeddings.dropout(embeddings)
        return embeddings

    def _calc_img_embeddings(self, image, bbox, position_ids):
        if image is not None:
            visual_embeddings = self.visual_act_fn(self.visual_proj(self.visual(image)))
        position_embeddings = self.embeddings.position_embeddings(position_ids)
        x1, y1, x2, y2, h, w = self.embeddings._calc_spatial_position_embeddings(bbox)
        if image is not None:
            embeddings = visual_embeddings + position_embeddings + x1 + y1 + x2 + y2 + w + h
        else:
            embeddings = position_embeddings + x1 + y1 + x2 + y2 + w + h
        if self.has_visual_segment_embedding:
            embeddings += self.visual_segment_embedding
        embeddings = self.visual_LayerNorm(embeddings)
        embeddings = self.visual_dropout(embeddings)
        return embeddings

    def _calc_visual_bbox(self, image_feature_pool_shape, bbox, device, final_shape):
        visual_bbox_x = torch.div(
            torch.arange(
                0,
                1000 * (image_feature_pool_shape[1] + 1),
                1000,
                device=device,
                dtype=bbox.dtype,
            ),
            self.config.image_feature_pool_shape[1],
            rounding_mode="floor",
        )
        visual_bbox_y = torch.div(
            torch.arange(
                0,
                1000 * (self.config.image_feature_pool_shape[0] + 1),
                1000,
                device=device,
                dtype=bbox.dtype,
            ),
            self.config.image_feature_pool_shape[0],
            rounding_mode="floor",
        )
        visual_bbox = torch.stack(
            [
                visual_bbox_x[:-1].repeat(image_feature_pool_shape[0], 1),
                visual_bbox_y[:-1].repeat(image_feature_pool_shape[1], 1).transpose(0, 1),
                visual_bbox_x[1:].repeat(image_feature_pool_shape[0], 1),
                visual_bbox_y[1:].repeat(image_feature_pool_shape[1], 1).transpose(0, 1),
            ],
            dim=-1,
        ).view(-1, bbox.size(-1))

        visual_bbox = visual_bbox.repeat(final_shape[0], 1, 1)

        return visual_bbox

    def get_input_embeddings(self):
        return self.embeddings.word_embeddings

    def resize_position_embeddings(self, new_num_position_embeddings):
        """
        Resizes position embeddings of the model if `new_num_position_embeddings != config.max_position_embeddings`.

        Arguments:
            new_num_position_embeddings (`int`):
            The number of new position embedding matrix. If position embeddings are learned, increasing the size
            will add newly initialized vectors at the end, whereas reducing the size will remove vectors from the
            end.
        """
        num_position_embeds_diff = new_num_position_embeddings - self.config.max_position_embeddings
        self.embeddings.position_ids = torch.arange(new_num_position_embeddings).expand((1, -1))

        # no resizing needs to be done if the length stays the same
        if num_position_embeds_diff == 0:
            return

        logger.info(
            f"Setting `config.max_position_embeddings={new_num_position_embeddings}`..."
        )
        self.config.max_position_embeddings = new_num_position_embeddings

        old_position_embeddings_weight = self.embeddings.position_embeddings.weight

        self.embeddings.position_embeddings = nn.Embedding(
            self.config.max_position_embeddings, self.config.hidden_size)

        with torch.no_grad():
            if num_position_embeds_diff > 0:
                self.embeddings.position_embeddings.weight[:-num_position_embeds_diff] = old_position_embeddings_weight
            else:
                self.embeddings.position_embeddings.weight = old_position_embeddings_weight[:num_position_embeds_diff]

    def forward(
            self,
            input_ids=None,
            bbox=None,
            image=None,
            token_type_ids=None,
            position_ids=None,
            attention_mask=None,
            head_mask=None,
            inputs_embeds=None,     # TODO：1
            output_hidden_states=False,
            output_attentions=False,
    ):
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )

        if input_ids is not None and inputs_embeds is not None:
            raise ValueError("You cannot specify both input_ids and inputs_embeds at the same time")
        elif input_ids is not None:
            input_shape = input_ids.size()
        elif inputs_embeds is not None:
            input_shape = inputs_embeds.size()[:-1]
        else:
            raise ValueError("You have to specify either input_ids or inputs_embeds")
        device = input_ids.device if input_ids is not None else inputs_embeds.device

        visual_shape = list(input_shape)
        visual_shape[1] = self.config.image_feature_pool_shape[0] * self.config.image_feature_pool_shape[1]
        visual_shape = torch.Size(visual_shape)
        final_shape = list(input_shape)
        final_shape[1] = final_shape[1] + visual_shape[1]
        final_shape = torch.Size(final_shape)

        visual_bbox = self._calc_visual_bbox(self.config.image_feature_pool_shape, bbox, device, final_shape)
        final_bbox = torch.cat([bbox, visual_bbox], dim=1)

        if attention_mask is None:
            attention_mask = torch.ones(input_shape, device=device)

        visual_attention_mask = torch.ones(visual_shape, device=device)
        final_attention_mask = torch.cat([attention_mask, visual_attention_mask], dim=1)

        if token_type_ids is None:
            token_type_ids = torch.zeros(input_shape, dtype=torch.long, device=device)

        if position_ids is None:
            seq_length = input_shape[1]
            position_ids = self.embeddings.position_ids[:, :seq_length]
            position_ids = position_ids.expand(input_shape)

        visual_position_ids = torch.arange(0, visual_shape[1], dtype=torch.long, device=device).repeat(
            input_shape[0], 1
        )
        final_position_ids = torch.cat([position_ids, visual_position_ids], dim=1)

        if bbox is None:
            bbox = torch.zeros(input_shape + [4], dtype=torch.long, device=device)

        text_layout_emb = self._calc_text_embeddings(
            input_ids=input_ids,
            bbox=bbox,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
        )

        visual_emb = self._calc_img_embeddings(
            image=image,
            bbox=visual_bbox,
            position_ids=visual_position_ids,
        )
        final_emb = torch.cat([text_layout_emb, visual_emb], dim=1)

        extended_attention_mask = final_attention_mask.unsqueeze(1).unsqueeze(2)
        extended_attention_mask = (1.0 - extended_attention_mask) * -10000.0
        if self.use_flash_attn:
            extended_attention_mask = extended_attention_mask.to(torch.bool)

        if head_mask is not None:
            if head_mask.dim() == 1:
                head_mask = head_mask.unsqueeze(0).unsqueeze(0).unsqueeze(
                    -1).unsqueeze(-1)
                head_mask = head_mask.expand(self.config.num_hidden_layers,
                                             -1, -1, -1, -1)
            elif head_mask.dim() == 2:
                head_mask = head_mask.unsqueeze(1).unsqueeze(-1).unsqueeze(-1)
        else:
            head_mask = [None] * self.config.num_hidden_layers

        encoder_outputs = self.encoder(
            final_emb,
            extended_attention_mask,
            bbox=final_bbox,
            position_ids=final_position_ids,
            head_mask=head_mask,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
        )
        sequence_output = encoder_outputs[0]
        pooled_output = self.pooler(sequence_output)
        return sequence_output, pooled_output
