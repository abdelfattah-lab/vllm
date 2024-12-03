from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Set, Type, Tuple
import torch
import torch.nn as nn

from vllm.attention.backends.abstract import (
    AttentionBackend, AttentionImpl, AttentionMetadata, 
    AttentionMetadataBuilder, AttentionType
)
from vllm.attention.backends.utils import (
    compute_slot_mapping, make_tensor_with_pad, is_all_encoder_attn_metadata_set
)

@dataclass
class PaluAttentionMetadata(AttentionMetadata):
    """Metadata for Palu compressed attention."""
    # Original seq lengths
    seq_lens: Optional[List[int]]
    # Seq lengths as tensor
    seq_lens_tensor: Optional[torch.Tensor]
    # Context lengths tensor
    context_lens_tensor: Optional[torch.Tensor]
    # Block tables for KV cache mapping
    block_tables: Optional[torch.Tensor]
    # Whether to use CUDA graph
    use_cuda_graph: bool
    # Maximum sequence length for prefill
    max_prefill_seq_len: int
    # Maximum sequence length for decode
    max_decode_seq_len: int

    @property
    def prefill_metadata(self) -> Optional["PaluAttentionMetadata"]:
        if self.num_prefills == 0:
            return None
        
        return PaluAttentionMetadata(
            num_prefills=self.num_prefills,
            num_prefill_tokens=self.num_prefill_tokens,
            num_decode_tokens=0,
            slot_mapping=self.slot_mapping[:self.num_prefill_tokens],
            seq_lens=self.seq_lens[:self.num_prefills] if self.seq_lens else None,
            seq_lens_tensor=self.seq_lens_tensor[:self.num_prefills] if self.seq_lens_tensor else None,
            context_lens_tensor=self.context_lens_tensor[:self.num_prefills] if self.context_lens_tensor else None,
            block_tables=self.block_tables[:self.num_prefills] if self.block_tables else None,
            use_cuda_graph=False,
            max_prefill_seq_len=self.max_prefill_seq_len,
            max_decode_seq_len=0,
            multi_modal_placeholder_index_maps=self.multi_modal_placeholder_index_maps
        )

    @property
    def decode_metadata(self) -> Optional["PaluAttentionMetadata"]:
        if self.num_decode_tokens == 0:
            return None

        return PaluAttentionMetadata(
            num_prefills=0,
            num_prefill_tokens=0,
            num_decode_tokens=self.num_decode_tokens,
            slot_mapping=self.slot_mapping[self.num_prefill_tokens:],
            seq_lens=None,
            seq_lens_tensor=self.seq_lens_tensor[self.num_prefills:] if self.seq_lens_tensor else None,
            context_lens_tensor=None,
            block_tables=self.block_tables[self.num_prefills:] if self.block_tables else None,
            use_cuda_graph=self.use_cuda_graph,
            max_prefill_seq_len=0,
            max_decode_seq_len=self.max_decode_seq_len,
            multi_modal_placeholder_index_maps=None
        )

class PaluAttentionBackend(AttentionBackend):
    """Backend for Palu compressed attention."""
    
    @staticmethod
    def get_name() -> str:
        return "PALU"

    @staticmethod
    def get_impl_cls() -> Type["AttentionImpl"]:
        return PaluAttentionImpl

    @staticmethod
    def get_metadata_cls() -> Type["AttentionMetadata"]:
        return PaluAttentionMetadata

    @staticmethod
    def get_builder_cls() -> Type["AttentionMetadataBuilder"]:
        return PaluAttentionMetadataBuilder

    @staticmethod
    def get_state_cls() -> Type["AttentionState"]:
        return CommonAttentionState

    @staticmethod
    def get_kv_cache_shape(
        num_blocks: int,
        block_size: int,
        num_kv_heads: int,
        head_size: int,
    ) -> Tuple[int, ...]:
        """Get shape for compressed KV cache."""
        if block_size % 16 != 0:
            raise ValueError("Block size must be multiple of 16")
            
        compressed_size = int(head_size * 0.5)  # 0.5 is compression rate
        return (2, num_blocks, block_size, num_kv_heads, compressed_size)

    @staticmethod
    def swap_blocks(
        src_kv_cache: torch.Tensor,
        dst_kv_cache: torch.Tensor,
        src_to_dst: torch.Tensor,
    ) -> None:
        """Swap blocks between source and destination KV caches."""
        for cache_id in [0, 1]:  # key and value caches
            src_cache = src_kv_cache[cache_id]
            dst_cache = dst_kv_cache[cache_id]
            dst_cache.index_copy_(0, src_to_dst, src_cache)

    @staticmethod
    def copy_blocks(
        kv_caches: List[torch.Tensor],
        src_to_dists: torch.Tensor,
    ) -> None:
        """Copy blocks between KV caches."""
        num_caches = len(kv_caches)
        for i in range(num_caches):
            for j in range(i + 1, num_caches):
                src_to_dist = src_to_dists[i][j]
                if src_to_dist.numel() == 0:
                    continue
                PaluAttentionBackend.swap_blocks(
                    kv_caches[i], kv_caches[j], src_to_dist)

class PaluAttentionImpl(AttentionImpl):
    """Implementation of Palu compressed attention."""
    def __init__(
        self,
        num_heads: int,
        head_size: int,
        scale: float,
        num_kv_heads: Optional[int] = None,
        alibi_slopes: Optional[List[float]] = None,
        sliding_window: Optional[int] = None,
        kv_cache_dtype: str = "auto",
        blocksparse_params: Optional[Dict[str, Any]] = None,
        logits_soft_cap: Optional[float] = None,
    ) -> None:
        super().__init__()
        self.num_heads = num_heads
        self.head_size = head_size
        self.scale = scale
        self.num_kv_heads = num_kv_heads or num_heads
        self.sliding_window = sliding_window
        self.kv_cache_dtype = kv_cache_dtype
        self.logits_soft_cap = logits_soft_cap
        
        # Palu-specific parameters
        self.compression_rate = 0.5  # Can be made configurable
        self.group_size = 4
        
        # Initialize projection matrices
        compressed_size = int(head_size * self.compression_rate)
        num_groups = num_kv_heads // self.group_size
        
        self.register_buffer('k_down_proj', 
                           torch.zeros(num_groups, 
                                     self.group_size * head_size, 
                                     self.group_size * compressed_size))
        self.register_buffer('v_down_proj',
                           torch.zeros(num_groups,
                                     self.group_size * head_size,
                                     self.group_size * compressed_size))
        self.register_buffer('k_up_proj',
                           torch.zeros(num_groups,
                                     self.group_size * compressed_size,
                                     self.group_size * head_size))
        self.register_buffer('v_up_proj',
                           torch.zeros(num_groups,
                                     self.group_size * compressed_size,
                                     self.group_size * head_size))
        
        self._initialize_projections()

    def _initialize_projections(self):
        """Initialize projection matrices with proper scaling."""
        with torch.no_grad():
            for proj in [self.k_down_proj, self.v_down_proj,
                        self.k_up_proj, self.v_up_proj]:
                nn.init.normal_(proj, mean=0.0, std=0.02)

    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        kv_cache: torch.Tensor,
        attn_metadata: PaluAttentionMetadata,
        k_scale: float = 1.0,
        v_scale: float = 1.0,
        attn_type: AttentionType = AttentionType.DECODER,
    ) -> torch.Tensor:
        """Forward pass with Palu compression."""
        batch_size, seq_len, num_heads, head_size = key.shape
        num_groups = num_heads // self.group_size
        
        # Reshape for group-wise operations
        key_grouped = key.view(batch_size, seq_len, num_groups, 
                             self.group_size * head_size)
        value_grouped = value.view(batch_size, seq_len, num_groups,
                                 self.group_size * head_size)
        
        # Project to compressed space
        key_compressed = torch.matmul(key_grouped, self.k_down_proj)
        value_compressed = torch.matmul(value_grouped, self.v_down_proj)
        
        # Update KV cache if needed
        if attn_metadata.is_prompt and kv_cache is not None:
            compressed_size = key_compressed.size(-1)
            kv_cache[..., :compressed_size] = key_compressed
            kv_cache[..., compressed_size:] = value_compressed
        
        # Reconstruct key and value
        key_full = torch.matmul(key_compressed, self.k_up_proj)
        value_full = torch.matmul(value_compressed, self.v_up_proj)
        
        # Reshape back
        key_full = key_full.view(batch_size, seq_len, num_heads, head_size)
        value_full = value_full.view(batch_size, seq_len, num_heads, head_size)
        
        # Regular attention computation
        attn_weights = torch.matmul(query, key_full.transpose(-2, -1))
        attn_weights = attn_weights * self.scale
        
        if self.sliding_window is not None:
            window_mask = self._get_window_mask(seq_len, self.sliding_window)
            attn_weights = attn_weights.masked_fill(~window_mask, float('-inf'))
            
        attn_weights = torch.softmax(attn_weights, dim=-1)
        
        if self.logits_soft_cap is not None:
            cap = self.logits_soft_cap
            attn_weights = (cap * attn_weights) / (cap + attn_weights)
            
        output = torch.matmul(attn_weights, value_full)
        return output

class PaluAttentionMetadataBuilder(AttentionMetadataBuilder):
    """Builder for Palu attention metadata."""

    def __init__(self, input_builder: "ModelRunnerInputBuilderBase") -> None:
        self.input_builder = input_builder
        self.runner = input_builder.runner
        self.sliding_window = input_builder.sliding_window
        self.block_size = input_builder.block_size
        
        # Initialize storage for metadata
        self.slot_mapping: List[int] = []
        self.seq_lens: List[int] = []
        self.context_lens: List[int] = []
        self.block_tables: List[List[int]] = []
        
        self.num_prefills = 0
        self.num_prefill_tokens = 0
        self.num_decode_tokens = 0

    def build(
        self, 
        seq_lens: List[int],
        query_lens: List[int],
        cuda_graph_pad_size: int,
        batch_size: int
    ) -> PaluAttentionMetadata:
        """Build attention metadata."""
        device = self.runner.device
        use_cuda_graph = cuda_graph_pad_size != -1
        
        max_prefill_seq_len = max(self.seq_lens[:self.num_prefills], default=0)
        max_decode_seq_len = max(self.seq_lens[self.num_prefills:], default=0)
        
        # Create tensors
        seq_lens_tensor = torch.tensor(seq_lens, dtype=torch.int32, device=device)
        context_lens_tensor = torch.tensor(self.context_lens, dtype=torch.int32, device=device)
        
        # Handle block tables
        if use_cuda_graph:
            # Pad block tables for CUDA graphs
            block_tables = torch.zeros(
                (batch_size, max(max_prefill_seq_len, max_decode_seq_len) // self.block_size),
                dtype=torch.int32,
                device=device
            )
            for i, table in enumerate(self.block_tables):
                if table:
                    block_tables[i, :len(table)] = torch.tensor(table, device=device)
        else:
            block_tables = make_tensor_with_pad(
                self.block_tables,
                pad=0,
                dtype=torch.int32,
                device=device
            )
            
        slot_mapping_tensor = torch.tensor(
            self.slot_mapping, dtype=torch.long, device=device)
            
        return PaluAttentionMetadata(
            num_prefills=self.num_prefills,
            num_prefill_tokens=self.num_prefill_tokens,
            num_decode_tokens=self.num_decode_tokens,
            slot_mapping=slot_mapping_tensor,
            seq_lens=self.seq_lens,
            seq_lens_tensor=seq_lens_tensor,
            context_lens_tensor=context_lens_tensor,
            block_tables=block_tables,
            use_cuda_graph=use_cuda_graph,
            max_prefill_seq_len=max_prefill_seq_len,
            max_decode_seq_len=max_decode_seq_len,
            multi_modal_placeholder_index_maps=None
        )