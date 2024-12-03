import unittest
from typing import List, Optional, Tuple

import torch

from vllm.attention import AttentionMetadata, AttentionType
from vllm.attention.backends.palu_attn import (
    PaluAttentionBackend, PaluAttentionImpl, PaluAttentionMetadata
)
from vllm.config import CacheConfig
from vllm.utils import StopSequences

class TestPaluAttention(unittest.TestCase):
    """Test suite for Palu attention implementation."""
    
    def setUp(self):
        """Setup common test parameters."""
        self.batch_size = 2
        self.seq_len = 32
        self.num_heads = 8
        self.head_size = 64
        self.num_kv_heads = 8
        self.scale = 1.0 / float(self.head_size) ** 0.5
        self.dtype = torch.float16
        self.device = "cuda"
        
        # Create test tensors
        self.query = torch.randn(
            self.batch_size, self.seq_len, 
            self.num_heads, self.head_size,
            dtype=self.dtype,
            device=self.device
        )
        self.key = torch.randn_like(self.query)
        self.value = torch.randn_like(self.query)
        
        # Create test metadata
        self.metadata = PaluAttentionMetadata(
            num_prefills=1,
            num_prefill_tokens=16,
            num_decode_tokens=16,
            slot_mapping=torch.arange(32, dtype=torch.long, device=self.device),
            seq_lens=[16, 16],
            seq_lens_tensor=torch.tensor([16, 16], 
                                       dtype=torch.int32, 
                                       device=self.device),
            context_lens_tensor=torch.tensor([16, 16], 
                                           dtype=torch.int32, 
                                           device=self.device),
            block_tables=torch.arange(4, dtype=torch.int32, 
                                    device=self.device).view(2, 2),
            use_cuda_graph=False,
            max_prefill_seq_len=16,
            max_decode_seq_len=16,
            multi_modal_placeholder_index_maps=None
        )

    def test_backend_creation(self):
        """Test basic backend creation and properties."""
        backend = PaluAttentionBackend()
        self.assertEqual(backend.get_name(), "PALU")
        self.assertEqual(backend.get_impl_cls(), PaluAttentionImpl)
        
        # Test KV cache shape
        shape = backend.get_kv_cache_shape(
            num_blocks=4,
            block_size=16,
            num_kv_heads=8,
            head_size=64
        )
        expected_shape = (2, 4, 16, 8, 32)  # 32 is compressed size (64 * 0.5)
        self.assertEqual(shape, expected_shape)

    def test_attention_impl(self):
        """Test attention implementation."""
        impl = PaluAttentionImpl(
            num_heads=self.num_heads,
            head_size=self.head_size,
            scale=self.scale,
            num_kv_heads=self.num_kv_heads,
        ).to(self.device)
        
        # Create compressed KV cache
        compressed_size = int(self.head_size * 0.5)
        kv_cache = torch.zeros(
            2,  # key and value
            4,  # num_blocks
            16,  # block_size
            self.num_kv_heads,
            compressed_size,
            dtype=self.dtype,
            device=self.device
        )
        
        # Forward pass
        output = impl.forward(
            query=self.query,
            key=self.key,
            value=self.value,
            kv_cache=kv_cache,
            attn_metadata=self.metadata
        )
        
        # Check output shape
        expected_shape = (self.batch_size, self.seq_len, 
                         self.num_heads, self.head_size)
        self.assertEqual(output.shape, expected_shape)
        
        # Check KV cache was updated
        self.assertFalse(torch.all(kv_cache == 0))

    def test_sliding_window(self):
        """Test attention with sliding window."""
        window_size = 16
        impl = PaluAttentionImpl(
            num_heads=self.num_heads,
            head_size=self.head_size,
            scale=self.scale,
            num_kv_heads=self.num_kv_heads,
            sliding_window=window_size
        ).to(self.device)
        
        # Create longer sequence to test window
        long_seq_len = 64
        query = torch.randn(
            self.batch_size, long_seq_len,
            self.num_heads, self.head_size,
            dtype=self.dtype,
            device=self.device
        )
        
        # Create compressed KV cache
        compressed_size = int(self.head_size * 0.5)
        kv_cache = torch.zeros(
            2, 8, 16, self.num_kv_heads, compressed_size,
            dtype=self.dtype,
            device=self.device
        )
        
        # Forward pass
        output = impl.forward(
            query=query,
            key=self.key,
            value=self.value,
            kv_cache=kv_cache,
            attn_metadata=self.metadata
        )
        
        # Check output shape
        expected_shape = (self.batch_size, long_seq_len, 
                         self.num_heads, self.head_size)
        self.assertEqual(output.shape, expected_shape)

    def test_memory_efficiency(self):
        """Test memory usage of compressed attention."""
        impl = PaluAttentionImpl(
            num_heads=self.num_heads,
            head_size=self.head_size,
            scale=self.scale,
            num_kv_heads=self.num_kv_heads,
        ).to(self.device)
        
        # Measure memory for uncompressed cache
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        
        uncompressed_cache = torch.zeros(
            2, 4, 16, self.num_kv_heads, self.head_size,
            dtype=self.dtype,
            device=self.device
        )
        uncompressed_mem = torch.cuda.max_memory_allocated()
        
        # Measure memory for compressed cache
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        
        compressed_size = int(self.head_size * 0.5)
        compressed_cache = torch.zeros(
            2, 4, 16, self.num_kv_heads, compressed_size,
            dtype=self.dtype,
            device=self.device
        )
        compressed_mem = torch.cuda.max_memory_allocated()
        
        # Check memory reduction
        self.assertLess(compressed_mem, uncompressed_mem)
        reduction = 1 - (compressed_mem / uncompressed_mem)
        self.assertGreaterEqual(reduction, 0.45)  # At least 45% reduction

    def test_attention_accuracy(self):
        """Test accuracy of compressed attention."""
        impl = PaluAttentionImpl(
            num_heads=self.num_heads,
            head_size=self.head_size,
            scale=self.scale,
            num_kv_heads=self.num_kv_heads,
        ).to(self.device)
        
        # Create reference implementation (uncompressed)
        ref_output = torch.nn.functional.scaled_dot_product_attention(
            self.query,
            self.key,
            self.value,
            scale=self.scale
        )
        
        # Run compressed attention
        compressed_size = int(self.head_size * 0.5)
        kv_cache = torch.zeros(
            2, 4, 16, self.num_kv_heads, compressed_size,
            dtype=self.dtype,
            device=self.device
        )
        
        output = impl.forward(
            query=self.query,
            key=self.key,
            value=self.value,
            kv_cache=kv_cache,
            attn_metadata=self.metadata
        )
        
        # Check relative error
        rel_error = torch.norm(output - ref_output) / torch.norm(ref_output)
        self.assertLess(rel_error, 0.1)  # Less than 10% relative error

    def test_cuda_graph_compatibility(self):
        """Test compatibility with CUDA graphs."""
        if not torch.cuda.is_available():
            self.skipTest("CUDA not available")
            
        impl = PaluAttentionImpl(
            num_heads=self.num_heads,
            head_size=self.head_size,
            scale=self.scale,
            num_kv_heads=self.num_kv_heads,
        ).to(self.device)
        
        # Create compressed KV cache
        compressed_size = int(self.head_size * 0.5)
        kv_cache = torch.zeros(
            2, 4, 16, self.num_kv_heads, compressed_size,
            dtype=self.dtype,
            device=self.device
        )
        
        # Enable CUDA graph
        self.metadata.use_cuda_graph = True
        
        # Capture graph
        s = torch.cuda.Stream()
        s.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(s):
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph):
                output = impl.forward(
                    query=self.query,
                    key=self.key,
                    value=self.value,
                    kv_cache=kv_cache,
                    attn_metadata=self.metadata
                )
        
        # Replay graph
        graph.replay()
        
        # Check output
        self.assertTrue(output is not None)
        self.assertEqual(output.shape, 
                        (self.batch_size, self.seq_len, 
                         self.num_heads, self.head_size))

if __name__ == '__main__':
    unittest.main(verbosity=2)