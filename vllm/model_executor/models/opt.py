# coding=utf-8
# Adapted from
# https://github.com/huggingface/transformers/blob/v4.28.0/src/transformers/models/opt/modeling_opt.py
# Copyright 2023 The vLLM team.
# Copyright 2022 The Fairseq Authors and The HuggingFace Inc. team. All rights
# reserved.
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
"""Inference-only OPT model compatible with HuggingFace weights."""
from typing import Iterable, List, Optional, Tuple, Union

import torch
from torch import nn
from transformers import OPTConfig
import torch.nn.functional as F

from vllm.attention import Attention, AttentionMetadata
from vllm.config import CacheConfig
from vllm.distributed import get_pp_group, get_tensor_model_parallel_world_size
from vllm.model_executor.layers.activation import get_act_fn
from vllm.model_executor.layers.linear import (ColumnParallelLinear,
                                               QKVParallelLinear,
                                               ReplicatedLinear,
                                               RowParallelLinear)
from vllm.model_executor.layers.logits_processor import LogitsProcessor
from vllm.model_executor.layers.quantization import QuantizationConfig
from vllm.model_executor.layers.sampler import Sampler, SamplerOutput
from vllm.model_executor.layers.vocab_parallel_embedding import (
    ParallelLMHead, VocabParallelEmbedding)
from vllm.model_executor.model_loader.weight_utils import default_weight_loader
from vllm.model_executor.sampling_metadata import SamplingMetadata
from vllm.sequence import IntermediateTensors

from .interfaces import SupportsPP
from .utils import (is_pp_missing_parameter,
                    make_empty_intermediate_tensors_factory, make_layers)

glob_n = 1024
glob_n_h = 2048
#range: 0 ~ 1
SPARSITY = 0.05

class OPTLearnedPositionalEmbedding(nn.Embedding):

    def __init__(self, num_embeddings: int, embedding_dim: int):
        # OPT is set up so that if padding_idx is specified then offset the
        # embedding ids by 2 and adjust num_embeddings appropriately. Other
        # models don't have this hack
        self.offset = 2
        super().__init__(num_embeddings + self.offset, embedding_dim)

    def forward(self, positions: torch.Tensor):
        return super().forward(positions + self.offset)


class OPTAttention(nn.Module):

    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        bias: bool = True,
        cache_config: Optional[CacheConfig] = None,
        quant_config: Optional[QuantizationConfig] = None,
    ) -> None:
        super().__init__()
        self.embed_dim = embed_dim
        tensor_model_parallel_world_size = (
            get_tensor_model_parallel_world_size())
        total_num_heads = num_heads
        assert num_heads % tensor_model_parallel_world_size == 0
        self.num_heads = total_num_heads // tensor_model_parallel_world_size
        self.head_dim = embed_dim // total_num_heads
        self.scaling = self.head_dim**-0.5

        self.qkv_proj = QKVParallelLinear(
            embed_dim,
            self.head_dim,
            total_num_heads,
            bias=bias,
            quant_config=quant_config,
        )
        self.out_proj = RowParallelLinear(
            embed_dim,
            embed_dim,
            bias=bias,
            quant_config=quant_config,
        )
        self.attn = Attention(self.num_heads,
                              self.head_dim,
                              scale=self.scaling,
                              cache_config=cache_config,
                              quant_config=quant_config)

    def forward(
        self,
        hidden_states: torch.Tensor,
        kv_cache: torch.Tensor,
        attn_metadata: AttentionMetadata,
        layer_head_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        qkv, _ = self.qkv_proj(hidden_states)
        q, k, v = qkv.chunk(chunks=3, dim=-1)
        attn_output = self.attn(q, k, v, kv_cache, attn_metadata)
        if layer_head_mask is not None:
            attn_output = attn_output.view(
                attn_output.size(0), self.num_heads, self.head_dim
            )
            attn_output = attn_output * layer_head_mask.view(-1, 1)
            attn_output = attn_output.view(attn_output.size(0), self.embed_dim)
        output, _ = self.out_proj(attn_output)
        return output


class OPTDecoderLayer(nn.Module):

    def __init__(
        self,
        config: OPTConfig,
        cache_config: Optional[CacheConfig] = None,
        quant_config: Optional[QuantizationConfig] = None,
    ):
        super().__init__()
        self.config = config
        self.embed_dim = config.hidden_size
        self.self_attn = OPTAttention(
            embed_dim=self.embed_dim,
            num_heads=config.num_attention_heads,
            bias=config.enable_bias,
            cache_config=cache_config,
            quant_config=quant_config,
        )
        self.do_layer_norm_before = config.do_layer_norm_before

        self.self_attn_layer_norm = nn.LayerNorm(
            self.embed_dim,
            elementwise_affine=config.layer_norm_elementwise_affine)
        self.fc1 = ColumnParallelLinear(
            self.embed_dim,
            config.ffn_dim,
            bias=config.enable_bias,
            quant_config=quant_config,
        )
        self.activation_fn = get_act_fn(config.activation_function,
                                        quant_config, config.ffn_dim)
        self.fc2 = RowParallelLinear(
            config.ffn_dim,
            self.embed_dim,
            bias=config.enable_bias,
            quant_config=quant_config,
        )
        self.final_layer_norm = nn.LayerNorm(
            self.embed_dim,
            elementwise_affine=config.layer_norm_elementwise_affine)

    def forward(
        self,
        hidden_states: torch.Tensor,
        kv_cache: torch.Tensor,
        attn_metadata: AttentionMetadata,
        layer_head_mask: Optional[torch.Tensor] = None,
        layer_fc_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        # Self Attention
        residual = hidden_states
        # 125m, 1.7B, ..., 175B applies layer norm BEFORE attention
        if self.do_layer_norm_before:
            hidden_states = self.self_attn_layer_norm(hidden_states)
        hidden_states = self.self_attn(hidden_states=hidden_states,
                                       kv_cache=kv_cache,
                                       attn_metadata=attn_metadata,
                                       layer_head_mask=layer_head_mask)
        hidden_states = residual + hidden_states
        # 350m applies layer norm AFTER attention
        if not self.do_layer_norm_before:
            hidden_states = self.self_attn_layer_norm(hidden_states)

        # Fully Connected
        residual = hidden_states
        # 125m, 1.7B, ..., 175B applies layer norm BEFORE attention
        if self.do_layer_norm_before:
            hidden_states = self.final_layer_norm(hidden_states)
        hidden_states, _ = self.fc1(hidden_states)
        if layer_fc_mask is not None:
            hidden_states = hidden_states * layer_fc_mask
        hidden_states = self.activation_fn(hidden_states)
        hidden_states, _ = self.fc2(hidden_states)
        hidden_states = residual + hidden_states
        # 350m applies layer norm AFTER attention
        if not self.do_layer_norm_before:
            hidden_states = self.final_layer_norm(hidden_states)
        return hidden_states


class OPTDecoder(nn.Module):

    def __init__(
        self,
        config: OPTConfig,
        cache_config: Optional[CacheConfig] = None,
        quant_config: Optional[QuantizationConfig] = None,
        prefix: str = "",
    ):
        super().__init__()
        self.config = config
        self.padding_idx = config.pad_token_id
        self.max_target_positions = config.max_position_embeddings
        self.vocab_size = config.vocab_size

        self.embed_tokens = VocabParallelEmbedding(
            config.vocab_size,
            config.word_embed_proj_dim,
        )
        # Positional embeddings are replicated (not sharded).
        self.embed_positions = OPTLearnedPositionalEmbedding(
            config.max_position_embeddings, config.hidden_size)

        # Project out & in will be replicated if they exist.
        if config.word_embed_proj_dim != config.hidden_size:
            self.project_out = ReplicatedLinear(config.hidden_size,
                                                config.word_embed_proj_dim,
                                                bias=False,
                                                quant_config=quant_config)
        else:
            self.project_out = None

        if config.word_embed_proj_dim != config.hidden_size:
            self.project_in = ReplicatedLinear(config.word_embed_proj_dim,
                                               config.hidden_size,
                                               bias=False,
                                               quant_config=quant_config)
        else:
            self.project_in = None

        # Note that the only purpose of config._remove_final_layer_norm is to
        # keep backward compatibility with checkpoints that have been fine-tuned
        # before transformers v4.20.1
        # see https://github.com/facebookresearch/metaseq/pull/164
        if config.do_layer_norm_before and not config._remove_final_layer_norm:
            self.final_layer_norm = nn.LayerNorm(
                config.hidden_size,
                elementwise_affine=config.layer_norm_elementwise_affine)
        else:
            self.final_layer_norm = None

        self.start_layer, self.end_layer, self.layers = make_layers(
            config.num_hidden_layers,
            lambda prefix: OPTDecoderLayer(config, cache_config, quant_config),
            prefix=f"{prefix}.layers")

    def get_input_embeddings(self, input_ids: torch.Tensor) -> torch.Tensor:
        return self.embed_tokens(input_ids)

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        kv_caches: List[torch.Tensor],
        attn_metadata: AttentionMetadata,
        intermediate_tensors: Optional[IntermediateTensors],
        head_mask: Optional[torch.Tensor] = None,
        fc_mask: Optional[torch.Tensor] = None,
        inputs_embeds: Optional[torch.Tensor] = None,
        return_first_layer_hidden_states: bool = False,
    ) -> Union[torch.Tensor, IntermediateTensors]:
        if get_pp_group().is_first_rank:
            if inputs_embeds is None:
                inputs_embeds = self.get_input_embeddings(input_ids)
            pos_embeds = self.embed_positions(positions)
            if self.project_in is not None:
                inputs_embeds, _ = self.project_in(inputs_embeds)
            hidden_states = inputs_embeds + pos_embeds
        else:
            assert intermediate_tensors is not None
            hidden_states = intermediate_tensors["hidden_states"]

        first_layer_hidden_states = None

        for i in range(self.start_layer, self.end_layer):
            layer = self.layers[i]
            layer_head_mask = head_mask[i] if head_mask is not None else None
            layer_fc_mask = fc_mask[i] if fc_mask is not None else None
            hidden_states = layer(hidden_states,
                                  kv_caches[i - self.start_layer],
                                  attn_metadata,
                                  layer_head_mask=layer_head_mask,
                                  layer_fc_mask=layer_fc_mask)
            if i == self.start_layer and return_first_layer_hidden_states:
                first_layer_hidden_states = hidden_states

        if not get_pp_group().is_last_rank:
            if return_first_layer_hidden_states:
                return IntermediateTensors({"hidden_states": hidden_states}), first_layer_hidden_states
            else:
                return IntermediateTensors({"hidden_states": hidden_states})
        if self.final_layer_norm is not None:
            hidden_states = self.final_layer_norm(hidden_states)
        if self.project_out is not None:
            hidden_states, _ = self.project_out(hidden_states)

        if return_first_layer_hidden_states:
            return hidden_states, first_layer_hidden_states
        else:
            return hidden_states


class OPTModel(nn.Module):

    def __init__(
        self,
        config: OPTConfig,
        cache_config: Optional[CacheConfig] = None,
        quant_config: Optional[QuantizationConfig] = None,
    ):
        super().__init__()
        self.decoder = OPTDecoder(config, cache_config, quant_config)
        self.make_empty_intermediate_tensors = (
            make_empty_intermediate_tensors_factory(["hidden_states"],
                                                    config.hidden_size))

    def get_input_embeddings(self, input_ids: torch.Tensor) -> torch.Tensor:
        return self.decoder.get_input_embeddings(input_ids)

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        kv_caches: List[torch.Tensor],
        attn_metadata: AttentionMetadata,
        intermediate_tensors: Optional[IntermediateTensors],
        head_mask: Optional[torch.Tensor] = None,
        fc_mask: Optional[torch.Tensor] = None,
        inputs_embeds: Optional[torch.Tensor] = None,
    ) -> Union[torch.Tensor, IntermediateTensors]:
        return self.decoder(input_ids,
                            positions,
                            kv_caches,
                            attn_metadata,
                            intermediate_tensors,
                            head_mask=head_mask,
                            fc_mask=fc_mask,
                            inputs_embeds=inputs_embeds)

    def get_first_layer_hidden_states(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        kv_caches: List[torch.Tensor],
        attn_metadata: AttentionMetadata,
        intermediate_tensors: Optional[IntermediateTensors] = None,
        inputs_embeds: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        outputs = self.decoder(input_ids,
                               positions,
                               kv_caches,
                               attn_metadata,
                               intermediate_tensors,
                               inputs_embeds=inputs_embeds,
                               return_first_layer_hidden_states=True)
                               
        hidden_states, first_layer_hidden_states = outputs
        return first_layer_hidden_states

class B1EPredModel(nn.Module):
    def __init__(self, embedding_dim=2048, output_dim=16):
        super(B1EPredModel, self).__init__()
        self.fc1 = nn.Linear(embedding_dim, glob_n_h)
        self.fc2 = nn.Linear(glob_n_h, output_dim)

    def forward(self, x):
        x = F.relu(self.fc1(x.squeeze()))
        x = self.fc2(x)
        return x

class B1EPredModelFFN(nn.Module):
    def __init__(self, embedding_dim=2048, output_dim=16):
        super(B1EPredModelFFN, self).__init__()
        self.fc1 = nn.Linear(embedding_dim, glob_n)
        self.fc2 = nn.Linear(glob_n, output_dim)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

def generate_head_mask(importances, sparsity, num_layers, num_heads):
    num_heads_total = importances.numel()
    num_to_keep = max(int((1 - sparsity) * num_heads_total), 1) 

    topk_values = torch.topk(importances, num_to_keep, largest=True).values
    cutoff = topk_values[-1] 

    mask = (importances >= cutoff).float().view(num_layers, num_heads)
    return mask.half()

def generate_fc_mask(importances, sparsity, num_layers, ffn_dim):
    num_neurons_total = importances.numel()
    num_to_keep = max(int((1 - sparsity) * num_neurons_total), 1)

    topk_values = torch.topk(importances, num_to_keep, largest=True).values
    cutoff = topk_values[-1] 

    mask = (importances >= cutoff).float().view(num_layers, ffn_dim)
    return mask.half()

class OPTForCausalLM(nn.Module, SupportsPP):

    # BitandBytes specific attributes
    bitsandbytes_stacked_params_mapping = {
        # shard_name, weight_name, index
        "q_proj": ("qkv_proj", 0),
        "k_proj": ("qkv_proj", 1),
        "v_proj": ("qkv_proj", 2),
    }
    default_bitsandbytes_target_modules = [
        ".q_proj.", ".k_proj.", ".v_proj.", ".out_proj.", ".fc1.", ".fc2."
    ]
    # in TP, these weights are partitioned along the column dimension (dim=-1)
    column_parallel_weights_modules = [".out_proj.", ".fc2."]

    def __init__(
        self,
        config: OPTConfig,
        cache_config: Optional[CacheConfig] = None,
        quant_config: Optional[QuantizationConfig] = None,
    ):
        super().__init__()
        self.config = config
        self.quant_config = quant_config
        self.model = OPTModel(config, cache_config, quant_config)
        if self.config.tie_word_embeddings:
            self.lm_head = self.model.decoder.embed_tokens
        else:
            self.lm_head = ParallelLMHead(config.vocab_size,
                                          config.word_embed_proj_dim)
        self.logits_processor = LogitsProcessor(config.vocab_size)
        self.sampler = Sampler()
        self.make_empty_intermediate_tensors = (
            self.model.make_empty_intermediate_tensors)
        
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        self.head_predictor = B1EPredModel(config.hidden_size, config.num_attention_heads * config.num_hidden_layers)
        self.head_predictor.load_state_dict(torch.load('b1e.pt'))
        self.head_predictor.to(device)
        self.head_predictor.eval()

        self.ffn_predictor = B1EPredModelFFN(config.hidden_size, config.ffn_dim * config.num_hidden_layers)
        self.ffn_predictor.load_state_dict(torch.load('b1e_fc1.pt'))
        self.ffn_predictor.to(device)
        self.ffn_predictor.eval()

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        kv_caches: List[torch.Tensor],
        attn_metadata: AttentionMetadata,
        intermediate_tensors: Optional[IntermediateTensors] = None,
        # head_mask: Optional[torch.Tensor] = None,
        # fc_mask: Optional[torch.Tensor] = None,
    ) -> Union[torch.Tensor, IntermediateTensors]:
        # print(f"shape of input_ids: {input_ids.shape}")
        if intermediate_tensors is not None:
            hidden_states = intermediate_tensors["hidden_states"]
        else:
            inputs_embeds = self.model.get_input_embeddings(input_ids)
            hidden_states = inputs_embeds
        
        first_layer_hidden_states = self.model.get_first_layer_hidden_states(
            input_ids,
            positions,
            kv_caches,
            attn_metadata,
            inputs_embeds=inputs_embeds,
        )

        if first_layer_hidden_states.dim() == 2:
            pred_input = first_layer_hidden_states[-1, :].unsqueeze(0)
        else:
            pred_input = first_layer_hidden_states[:, -1, :]

        pred_input = (pred_input - pred_input.min()) / (pred_input.max() - pred_input.min() + 1e-6)

        head_importance_scores = self.head_predictor(pred_input)
        # if head_importance_scores.dim() > 1:
        #     head_importance_scores = head_importance_scores.mean(dim=0)
        head_importance_scores = head_importance_scores.view(-1)
        head_mask = generate_head_mask(head_importance_scores, SPARSITY, num_layers=self.config.num_hidden_layers, num_heads=self.config.num_attention_heads)
        # head_mask = head_mask.view(self.config.num_hidden_layers, self.config.num_attention_heads)
        head_mask[0, :] = 1.
        # print(f"Generated head_mask shape: {head_mask.shape}")

        ffn_importance_scores = self.ffn_predictor(pred_input)
        ffn_importance_scores = ffn_importance_scores.view(-1)
        # if head_importance_scores.dim() > 1:
        #     ffn_importance_scores = ffn_importance_scores.mean(dim=0)
        fc_mask = generate_fc_mask(ffn_importance_scores, SPARSITY, num_layers=self.config.num_hidden_layers, ffn_dim=self.config.ffn_dim)
        # fc_mask = fc_mask.view(self.config.num_hidden_layers, self.config.ffn_dim)
        fc_mask[0, :] = 1.
        # print(f"Generated fc_mask shape: {fc_mask.shape}")
        
        hidden_states = self.model(input_ids, positions, kv_caches,
                                   attn_metadata, intermediate_tensors,
                                   head_mask=head_mask, fc_mask=fc_mask)
        return hidden_states

    def compute_logits(
        self,
        hidden_states: torch.Tensor,
        sampling_metadata: SamplingMetadata,
    ) -> Optional[torch.Tensor]:
        logits = self.logits_processor(self.lm_head, hidden_states,
                                       sampling_metadata)
        return logits

    def sample(
        self,
        logits: torch.Tensor,
        sampling_metadata: SamplingMetadata,
    ) -> Optional[SamplerOutput]:
        next_tokens = self.sampler(logits, sampling_metadata)
        return next_tokens

    def load_weights(self, weights: Iterable[Tuple[str, torch.Tensor]]):
        stacked_params_mapping = [
            # (param_name, shard_name, shard_id)
            ("qkv_proj", "q_proj", "q"),
            ("qkv_proj", "k_proj", "k"),
            ("qkv_proj", "v_proj", "v"),
        ]
        params_dict = dict(self.named_parameters(remove_duplicate=False))
        for name, loaded_weight in weights:
            if "lm_head.weight" in name and self.config.tie_word_embeddings:
                continue
            if name.startswith("decoder."):
                name = "model." + name

            for (param_name, weight_name, shard_id) in stacked_params_mapping:
                if weight_name not in name:
                    continue
                name = name.replace(weight_name, param_name)
                # Skip loading extra bias for GPTQ models.
                if name.endswith(".bias") and name not in params_dict:
                    continue
                if is_pp_missing_parameter(name, self):
                    continue
                param = params_dict[name]
                weight_loader = param.weight_loader
                weight_loader(param, loaded_weight, shard_id)
                break
            else:
                # Skip loading extra bias for GPTQ models.
                if name.endswith(".bias") and name not in params_dict:
                    continue
                if is_pp_missing_parameter(name, self):
                    continue
                param = params_dict[name]
                weight_loader = getattr(param, "weight_loader",
                                        default_weight_loader)
                weight_loader(param, loaded_weight)