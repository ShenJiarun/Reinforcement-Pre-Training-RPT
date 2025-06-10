"""
Scaling Utilities for RPT

This module provides utilities for scaling RPT training to large models
and datasets, including distributed training and memory optimization.
"""

import torch
import torch.distributed as dist
from typing import Dict, List, Optional, Union, Any, Tuple
import logging
import os
import psutil
import GPUtil
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class ScalingConfig:
    """Configuration for scaling RPT training."""
    
    # Distributed training
    use_distributed: bool = False
    world_size: int = 1
    rank: int = 0
    local_rank: int = 0
    backend: str = "nccl"
    
    # Memory optimization
    gradient_checkpointing: bool = False
    use_deepspeed: bool = False
    cpu_offload: bool = False
    max_memory_per_gpu: Optional[float] = None
    
    # Compute optimization
    compile_model: bool = False
    use_flash_attention: bool = False
    mixed_precision: bool = True
    
    # Batch sizing
    auto_batch_size: bool = False
    max_batch_size: int = 32
    min_batch_size: int = 1


class ScalingUtils:
    """
    Utilities for scaling RPT training across multiple devices and optimizing memory usage.
    """
    
    def __init__(self, config: Optional[ScalingConfig] = None):
        """
        Initialize scaling utilities.
        
        Args:
            config: Scaling configuration
        """
        self.config = config or ScalingConfig()
        self.is_distributed = self.config.use_distributed and torch.cuda.device_count() > 1
        
    def setup_distributed_training(self) -> bool:
        """
        Setup distributed training environment.
        
        Returns:
            True if distributed training is available and setup
        """
        if not self.config.use_distributed:
            return False
            
        if not torch.cuda.is_available():
            logger.warning("CUDA not available, cannot use distributed training")
            return False
            
        if torch.cuda.device_count() <= 1:
            logger.warning("Only one GPU available, distributed training not needed")
            return False
            
        # Initialize process group
        if not dist.is_initialized():
            try:
                # Check for environment variables set by torchrun/launch utilities
                if "RANK" in os.environ and "WORLD_SIZE" in os.environ:
                    self.config.rank = int(os.environ["RANK"])
                    self.config.world_size = int(os.environ["WORLD_SIZE"])
                    self.config.local_rank = int(os.environ.get("LOCAL_RANK", 0))
                    
                    dist.init_process_group(
                        backend=self.config.backend,
                        rank=self.config.rank,
                        world_size=self.config.world_size
                    )
                    
                    torch.cuda.set_device(self.config.local_rank)
                    logger.info(f"Distributed training setup: rank {self.config.rank}/{self.config.world_size}")
                    return True
                else:
                    logger.warning("Environment variables for distributed training not found")
                    return False
                    
            except Exception as e:
                logger.error(f"Failed to initialize distributed training: {e}")
                return False
        
        return True
    
    def wrap_model_for_distributed(self, model: torch.nn.Module) -> torch.nn.Module:
        """
        Wrap model for distributed training.
        
        Args:
            model: Model to wrap
            
        Returns:
            Wrapped model
        """
        if not self.is_distributed:
            return model
            
        if dist.is_initialized():
            from torch.nn.parallel import DistributedDataParallel as DDP
            model = DDP(
                model,
                device_ids=[self.config.local_rank],
                output_device=self.config.local_rank,
                find_unused_parameters=True
            )
            logger.info("Model wrapped with DistributedDataParallel")
            
        return model
    
    def optimize_model_memory(self, model: torch.nn.Module) -> torch.nn.Module:
        """
        Apply memory optimizations to the model.
        
        Args:
            model: Model to optimize
            
        Returns:
            Optimized model
        """
        # Gradient checkpointing
        if self.config.gradient_checkpointing:
            if hasattr(model, 'gradient_checkpointing_enable'):
                model.gradient_checkpointing_enable()
                logger.info("Gradient checkpointing enabled")
            else:
                logger.warning("Model does not support gradient checkpointing")
        
        # Model compilation (PyTorch 2.0+)
        if self.config.compile_model:
            try:
                if hasattr(torch, 'compile'):
                    model = torch.compile(model)
                    logger.info("Model compiled with torch.compile")
                else:
                    logger.warning("torch.compile not available")
            except Exception as e:
                logger.warning(f"Model compilation failed: {e}")
        
        return model
    
    def get_optimal_batch_size(
        self,
        model: torch.nn.Module,
        sample_input: Dict[str, torch.Tensor],
        device: torch.device
    ) -> int:
        """
        Find optimal batch size through binary search.
        
        Args:
            model: Model to test
            sample_input: Sample input for testing
            device: Training device
            
        Returns:
            Optimal batch size
        """
        if not self.config.auto_batch_size:
            return self.config.max_batch_size
            
        logger.info("Finding optimal batch size...")
        
        model.train()
        optimal_batch_size = self.config.min_batch_size
        
        for batch_size in range(self.config.min_batch_size, self.config.max_batch_size + 1, 2):
            try:
                # Create test batch
                test_batch = {}
                for key, value in sample_input.items():
                    if torch.is_tensor(value):
                        # Repeat to create batch
                        test_batch[key] = value.repeat(batch_size, *[1] * (value.dim() - 1))
                    else:
                        test_batch[key] = value
                
                # Move to device
                test_batch = {k: v.to(device) if torch.is_tensor(v) else v for k, v in test_batch.items()}
                
                # Test forward pass
                with torch.cuda.amp.autocast(enabled=self.config.mixed_precision):
                    outputs = model(**test_batch)
                    
                    # Test backward pass
                    if isinstance(outputs, dict) and "logits" in outputs:
                        loss = outputs["logits"].sum()
                    else:
                        loss = outputs.sum() if hasattr(outputs, 'sum') else torch.tensor(0.0)
                        
                    loss.backward()
                
                # Clear gradients and cache
                model.zero_grad()
                torch.cuda.empty_cache()
                
                optimal_batch_size = batch_size
                logger.info(f"Batch size {batch_size} successful")
                
            except torch.cuda.OutOfMemoryError:
                logger.info(f"Batch size {batch_size} caused OOM, stopping search")
                torch.cuda.empty_cache()
                break
            except Exception as e:
                logger.warning(f"Error testing batch size {batch_size}: {e}")
                break
        
        logger.info(f"Optimal batch size found: {optimal_batch_size}")
        return optimal_batch_size
    
    def get_system_info(self) -> Dict[str, Any]:
        """
        Get comprehensive system information for scaling decisions.
        
        Returns:
            System information dictionary
        """
        info = {
            "cpu_count": psutil.cpu_count(),
            "cpu_usage": psutil.cpu_percent(),
            "memory_total": psutil.virtual_memory().total / (1024**3),  # GB
            "memory_available": psutil.virtual_memory().available / (1024**3),  # GB
            "memory_usage": psutil.virtual_memory().percent,
        }
        
        # GPU information
        if torch.cuda.is_available():
            info["cuda_available"] = True
            info["cuda_device_count"] = torch.cuda.device_count()
            info["cuda_current_device"] = torch.cuda.current_device()
            
            # GPU details
            try:
                gpus = GPUtil.getGPUs()
                info["gpu_details"] = []
                for gpu in gpus:
                    info["gpu_details"].append({
                        "id": gpu.id,
                        "name": gpu.name,
                        "memory_total": gpu.memoryTotal,
                        "memory_used": gpu.memoryUsed,
                        "memory_free": gpu.memoryFree,
                        "load": gpu.load,
                        "temperature": gpu.temperature
                    })
            except Exception as e:
                logger.warning(f"Could not get GPU details: {e}")
                info["gpu_details"] = []
        else:
            info["cuda_available"] = False
            info["cuda_device_count"] = 0
        
        # PyTorch information
        info["torch_version"] = torch.__version__
        info["torch_distributed_available"] = dist.is_available()
        
        return info
    
    def estimate_memory_usage(
        self,
        model: torch.nn.Module,
        batch_size: int,
        sequence_length: int,
        vocab_size: int = 50257,
        dtype: torch.dtype = torch.float32
    ) -> Dict[str, float]:
        """
        Estimate memory usage for training.
        
        Args:
            model: Model to analyze
            batch_size: Training batch size
            sequence_length: Input sequence length
            vocab_size: Vocabulary size
            dtype: Data type for calculations
            
        Returns:
            Memory usage estimates in GB
        """
        # Calculate bytes per element
        bytes_per_element = torch.tensor(0, dtype=dtype).element_size()
        
        # Model parameters
        param_count = sum(p.numel() for p in model.parameters())
        param_memory = param_count * bytes_per_element / (1024**3)
        
        # Gradients (same size as parameters)
        grad_memory = param_memory
        
        # Optimizer states (assume Adam: 2x parameters)
        optimizer_memory = 2 * param_memory
        
        # Activations (rough estimate)
        # Input: batch_size * sequence_length * hidden_size
        if hasattr(model, 'config') and hasattr(model.config, 'hidden_size'):
            hidden_size = model.config.hidden_size
        else:
            hidden_size = 768  # Default estimate
            
        activation_elements = batch_size * sequence_length * hidden_size
        activation_memory = activation_elements * bytes_per_element / (1024**3)
        
        # Attention matrices: batch_size * num_heads * seq_len * seq_len
        if hasattr(model, 'config') and hasattr(model.config, 'num_attention_heads'):
            num_heads = model.config.num_attention_heads
        else:
            num_heads = 12  # Default estimate
            
        attention_elements = batch_size * num_heads * sequence_length * sequence_length
        attention_memory = attention_elements * bytes_per_element / (1024**3)
        
        # Output logits: batch_size * sequence_length * vocab_size
        logit_elements = batch_size * sequence_length * vocab_size
        logit_memory = logit_elements * bytes_per_element / (1024**3)
        
        total_memory = (
            param_memory + grad_memory + optimizer_memory + 
            activation_memory + attention_memory + logit_memory
        )
        
        return {
            "parameters": param_memory,
            "gradients": grad_memory,
            "optimizer_states": optimizer_memory,
            "activations": activation_memory,
            "attention": attention_memory,
            "logits": logit_memory,
            "total_estimated": total_memory,
            "safety_factor": total_memory * 1.5  # 50% safety margin
        }
    
    def cleanup_distributed(self) -> None:
        """Clean up distributed training."""
        if dist.is_initialized():
            dist.destroy_process_group()
            logger.info("Distributed training cleanup completed")
    
    def get_scaling_recommendations(
        self,
        model: torch.nn.Module,
        target_batch_size: int,
        sequence_length: int
    ) -> Dict[str, Any]:
        """
        Get recommendations for scaling training.
        
        Args:
            model: Model to analyze
            target_batch_size: Desired batch size
            sequence_length: Input sequence length
            
        Returns:
            Scaling recommendations
        """
        system_info = self.get_system_info()
        memory_estimates = self.estimate_memory_usage(model, target_batch_size, sequence_length)
        
        recommendations = {
            "system_info": system_info,
            "memory_estimates": memory_estimates,
            "recommendations": []
        }
        
        # Memory recommendations
        available_memory = sum(gpu["memory_total"] for gpu in system_info.get("gpu_details", []))
        required_memory = memory_estimates["safety_factor"]
        
        if required_memory > available_memory:
            recommendations["recommendations"].append({
                "type": "memory",
                "issue": "Insufficient GPU memory",
                "suggestion": "Consider gradient checkpointing, smaller batch size, or CPU offloading"
            })
        
        # Distributed training recommendations
        if system_info["cuda_device_count"] > 1 and not self.config.use_distributed:
            recommendations["recommendations"].append({
                "type": "distributed",
                "issue": "Multiple GPUs available but not using distributed training",
                "suggestion": "Enable distributed training to utilize all GPUs"
            })
        
        # Optimization recommendations
        if not self.config.mixed_precision:
            recommendations["recommendations"].append({
                "type": "optimization",
                "issue": "Mixed precision not enabled",
                "suggestion": "Enable mixed precision training to reduce memory usage and increase speed"
            })
        
        return recommendations