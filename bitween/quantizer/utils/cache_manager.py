import torch
import os
import tempfile
import shutil
import gc
from typing import Dict, List
from tqdm import tqdm


class CacheManager:
    """
    Static class to handle caching of intermediate activations for quantization.
    
    Provides both memory and disk-based caching with memory limits and chunking support.
    """
    
    @staticmethod
    def cache_block_inputs(model, calib_data, block_names: List[str], nsamples: int,
                          cache_to_disk: bool = False, max_memory_mb: int = 512) -> Dict[str, any]:
        """
        Cache intermediate activations with memory-efficient disk storage.
        
        Args:
            model: The model to analyze
            calib_data: Calibration dataset
            block_names: List of block names to cache inputs for
            nsamples: Number of samples to process
            cache_to_disk: Whether to use disk caching
            max_memory_mb: Maximum memory limit in MB
            
        Returns:
            Dictionary mapping block names to cache file paths or memory cache keys
        """
        # Setup cache storage
        if cache_to_disk:
            cache_dir = tempfile.mkdtemp(prefix="bitween_cache_")
            cached_paths = {'_cache_dir': cache_dir}
        else:
            cached_inputs = {name: [] for name in block_names}
        
        # Memory tracking
        current_memory_mb = 0
        max_memory_bytes = max_memory_mb * 1024 * 1024
        
        # Initialize block-specific storage and chunk tracking
        block_caches = {name: [] for name in block_names}
        block_chunk_counters = {name: 0 for name in block_names}
        
        def make_hook(block_name):
            def hook_fn(module, input, output):
                nonlocal current_memory_mb
                
                # Extract input tensor
                if isinstance(input, tuple):
                    tensor = input[0].detach().cpu()
                else:
                    tensor = input.detach().cpu()
                
                # Calculate tensor size in bytes
                tensor_size = tensor.numel() * tensor.element_size()
                
                if cache_to_disk:
                    # Check memory limit before adding new tensor
                    if current_memory_mb + tensor_size > max_memory_bytes:
                        # Flush ALL blocks to disk and clear memory
                        for b_name, tensors in block_caches.items():
                            if tensors:  # Only flush non-empty caches
                                CacheManager._flush_block_cache_to_disk(
                                    cache_dir, b_name, tensors, block_chunk_counters[b_name]
                                )
                                block_chunk_counters[b_name] += 1
                        
                        # Clear all block caches and reset memory counter
                        for b_name in block_caches:
                            block_caches[b_name] = []
                        current_memory_mb = 0
                        
                        # Force garbage collection
                        gc.collect()
                        torch.cuda.empty_cache() if torch.cuda.is_available() else None
                        
                        print(f"Flushed cache chunks to disk, freed memory (limit: {max_memory_mb}MB)")
                    
                    block_caches[block_name].append(tensor)
                    current_memory_mb += tensor_size
                else:
                    # Direct memory storage
                    cached_inputs[block_name].append(tensor)
                    
            return hook_fn
        
        # Register hooks for each block
        hooks = []
        for block_name in block_names:
            block = CacheManager._get_module(model, block_name)
            hook = block.register_forward_hook(make_hook(block_name))
            hooks.append(hook)
        
        # Run calibration data through model to collect inputs
        model.eval()
        with torch.no_grad():
            dataloader = calib_data.get_dataloader(batch_size=1)
            
            for i, batch in enumerate(tqdm(dataloader, desc="Caching block inputs", total=nsamples)):
                if i >= nsamples:
                    break
                    
                input_ids = batch['input_ids'].to(model.device)
                
                # Forward pass to trigger hooks
                try:
                    _ = model(input_ids)
                except Exception as e:
                    print(f"Warning: Forward pass failed for batch {i}: {e}")
                    continue
                
                # Periodic memory cleanup during caching
                if cache_to_disk and (i + 1) % 50 == 0:
                    torch.cuda.empty_cache() if torch.cuda.is_available() else None
                    gc.collect()
        
        # Remove hooks
        for hook in hooks:
            hook.remove()
        
        # Finalize caching
        if cache_to_disk:
            # Flush any remaining data to disk
            for block_name in block_names:
                if block_caches[block_name]:
                    CacheManager._flush_block_cache_to_disk(
                        cache_dir, block_name, block_caches[block_name], block_chunk_counters[block_name]
                    )
                    block_chunk_counters[block_name] += 1
            
            # Clear all memory caches
            block_caches.clear()
            gc.collect()
            torch.cuda.empty_cache() if torch.cuda.is_available() else None
            
            # Return chunk info for each block
            total_size_mb = 0
            
            for block_name in block_names:
                # Collect all chunk files for this block
                chunk_files = []
                for chunk_id in range(block_chunk_counters[block_name]):
                    chunk_file = os.path.join(cache_dir, f"{block_name.replace('.', '_')}_chunk_{chunk_id}.pt")
                    if os.path.exists(chunk_file):
                        chunk_files.append(chunk_file)
                        total_size_mb += os.path.getsize(chunk_file) / (1024 * 1024)
                
                if chunk_files:
                    cached_paths[block_name] = {
                        'chunk_files': chunk_files,
                        'num_chunks': len(chunk_files)
                    }
                else:
                    print(f"Warning: No cache chunks found for block {block_name}")
            
            print(f"Cached {len(block_names)} blocks to disk ({total_size_mb:.1f}MB total in chunks)")
            return cached_paths
        else:
            print(f"Cached inputs for {len(block_names)} blocks in memory")
            return cached_inputs
    
    @staticmethod
    def _flush_block_cache_to_disk(cache_dir: str, block_name: str, tensors: List[torch.Tensor], chunk_id: int):
        """Save a block's cached tensors to disk as a separate chunk file."""
        if not tensors:
            return
            
        # Create unique chunk file name
        chunk_file = os.path.join(cache_dir, f"{block_name.replace('.', '_')}_chunk_{chunk_id}.pt")
        
        # Save directly to chunk file (no appending needed)
        torch.save(tensors, chunk_file)
    
    @staticmethod
    def load_block_cache(cache_path_or_data, block_name: str, cache_to_disk: bool = False) -> List[torch.Tensor]:
        """Load cached tensors for a specific block from chunk files."""
        if cache_to_disk:
            if isinstance(cache_path_or_data, dict) and block_name in cache_path_or_data:
                chunk_info = cache_path_or_data[block_name]
                
                if isinstance(chunk_info, dict) and 'chunk_files' in chunk_info:
                    # New chunked format
                    chunk_files = chunk_info['chunk_files']
                    
                    # Force garbage collection before loading
                    gc.collect()
                    torch.cuda.empty_cache() if torch.cuda.is_available() else None
                    
                    # Load all chunks and combine
                    all_tensors = []
                    for chunk_file in chunk_files:
                        if os.path.exists(chunk_file):
                            chunk_tensors = torch.load(chunk_file, map_location='cpu')
                            all_tensors.extend(chunk_tensors)
                        else:
                            print(f"Warning: Chunk file not found: {chunk_file}")
                    return all_tensors
                else:
                    print(f"Warning: Invalid cache format for block {block_name}")
                    return []
            else:
                return []
        else:
            # Direct memory access
            return cache_path_or_data.get(block_name, [])
    
    @staticmethod
    def cleanup_block_cache(cache_data, block_name: str, cache_to_disk: bool = False):
        """Clean up cache for a specific block (handles both chunked and single file formats)."""
        if cache_to_disk:
            if isinstance(cache_data, dict) and block_name in cache_data:
                chunk_info = cache_data[block_name]
                
                if isinstance(chunk_info, dict) and 'chunk_files' in chunk_info:
                    # New chunked format - delete all chunk files
                    chunk_files = chunk_info['chunk_files']
                    
                    for chunk_file in chunk_files:
                        if os.path.exists(chunk_file):
                            try:
                                os.remove(chunk_file)
                            except Exception as e:
                                print(f"Warning: Failed to delete chunk file {chunk_file}: {e}")
                
                # Remove from dict to prevent re-access
                del cache_data[block_name]
        else:
            # Clear from memory
            if block_name in cache_data:
                del cache_data[block_name]
        
        # Aggressive garbage collection
        gc.collect()
        torch.cuda.empty_cache() if torch.cuda.is_available() else None
    
    @staticmethod
    def cleanup_all_cache(cache_data, cache_to_disk: bool = False):
        """Clean up all cache files and directories."""
        if cache_to_disk and isinstance(cache_data, dict) and '_cache_dir' in cache_data:
            cache_dir = cache_data['_cache_dir']
            if cache_dir and os.path.exists(cache_dir):
                shutil.rmtree(cache_dir)
                print(f"Cleaned up cache directory: {cache_dir}")
    
    @staticmethod
    def _get_module(model, module_name: str):
        """Get a module by its dotted name."""
        tokens = module_name.split('.')
        cur_mod = model
        for token in tokens:
            cur_mod = getattr(cur_mod, token)
        return cur_mod