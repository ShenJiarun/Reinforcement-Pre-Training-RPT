"""
Distributed Training Support for Reinforcement Pre-Training

Enables multi-GPU and multi-node training with proper synchronization
of rewards, gradients, and model parameters across distributed processes.
"""

import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler
import os
import logging
from typing import Optional, Dict, Any, List
import socket
from contextlib import contextmanager

logger = logging.getLogger(__name__)


class DistributedTrainingManager:
    """
    Manages distributed training setup and coordination for RPT.
    """
    
    def __init__(
        self,
        backend: str = "nccl",
        init_method: str = "env://",
        world_size: Optional[int] = None,
        rank: Optional[int] = None
    ):
        """
        Initialize distributed training manager.
        
        Args:
            backend: Distributed backend ('nccl', 'gloo', 'mpi')
            init_method: Initialization method for process group
            world_size: Total number of processes
            rank: Rank of current process
        """
        self.backend = backend
        self.init_method = init_method
        self.world_size = world_size or int(os.environ.get('WORLD_SIZE', 1))
        self.rank = rank or int(os.environ.get('RANK', 0))
        self.local_rank = int(os.environ.get('LOCAL_RANK', 0))
        
        self.is_distributed = self.world_size > 1
        self.is_initialized = False
        
    def setup(self):
        """Initialize the distributed process group"""
        if not self.is_distributed:
            logger.info("Single GPU/CPU training - no distributed setup needed")
            return
            
        if self.is_initialized:
            logger.warning("Distributed training already initialized")
            return
            
        # Set CUDA device for this process
        if torch.cuda.is_available():
            torch.cuda.set_device(self.local_rank)
            
        # Initialize process group
        dist.init_process_group(
            backend=self.backend,
            init_method=self.init_method,
            world_size=self.world_size,
            rank=self.rank
        )
        
        self.is_initialized = True
        
        logger.info(f"Distributed training initialized:")
        logger.info(f"  World size: {self.world_size}")
        logger.info(f"  Rank: {self.rank}")
        logger.info(f"  Local rank: {self.local_rank}")
        logger.info(f"  Backend: {self.backend}")
        
    def cleanup(self):
        """Clean up distributed training"""
        if self.is_distributed and self.is_initialized:
            dist.destroy_process_group()
            self.is_initialized = False
            logger.info("Distributed training cleaned up")
    
    def wrap_model(self, model: torch.nn.Module) -> torch.nn.Module:
        """Wrap model for distributed training"""
        if not self.is_distributed:
            return model
            
        if torch.cuda.is_available():
            model = model.cuda(self.local_rank)
            model = DDP(model, device_ids=[self.local_rank])
        else:
            model = DDP(model)
            
        logger.info("Model wrapped with DistributedDataParallel")
        return model
    
    def create_distributed_sampler(self, dataset, shuffle: bool = True):
        """Create distributed sampler for dataset"""
        if not self.is_distributed:
            return None
            
        return DistributedSampler(
            dataset,
            num_replicas=self.world_size,
            rank=self.rank,
            shuffle=shuffle
        )
    
    @contextmanager
    def synchronized_execution(self):
        """Context manager for synchronized execution across processes"""
        if self.is_distributed:
            dist.barrier()
        yield
        if self.is_distributed:
            dist.barrier()
    
    def all_reduce_tensor(self, tensor: torch.Tensor, op=dist.ReduceOp.SUM) -> torch.Tensor:
        """Perform all-reduce operation on tensor"""
        if not self.is_distributed:
            return tensor
            
        # Clone tensor to avoid in-place modification
        reduced_tensor = tensor.clone()
        dist.all_reduce(reduced_tensor, op=op)
        
        return reduced_tensor
    
    def all_gather_tensor(self, tensor: torch.Tensor) -> List[torch.Tensor]:
        """Gather tensor from all processes"""
        if not self.is_distributed:
            return [tensor]
            
        tensor_list = [torch.zeros_like(tensor) for _ in range(self.world_size)]
        dist.all_gather(tensor_list, tensor)
        
        return tensor_list
    
    def broadcast_tensor(self, tensor: torch.Tensor, src: int = 0) -> torch.Tensor:
        """Broadcast tensor from source process to all processes"""
        if not self.is_distributed:
            return tensor
            
        dist.broadcast(tensor, src=src)
        return tensor
    
    def reduce_rewards(self, rewards: torch.Tensor) -> torch.Tensor:
        """
        Reduce rewards across all processes for consistent training.
        
        Args:
            rewards: Local rewards tensor
            
        Returns:
            Averaged rewards across all processes
        """
        if not self.is_distributed:
            return rewards
            
        # All-reduce and average
        total_rewards = self.all_reduce_tensor(rewards)
        avg_rewards = total_rewards / self.world_size
        
        return avg_rewards
    
    def sync_model_parameters(self, model: torch.nn.Module):
        """Synchronize model parameters across processes"""
        if not self.is_distributed:
            return
            
        for param in model.parameters():
            if param.requires_grad:
                self.all_reduce_tensor(param.data)
                param.data /= self.world_size
    
    def is_main_process(self) -> bool:
        """Check if this is the main process (rank 0)"""
        return self.rank == 0
    
    def print_rank0(self, message: str):
        """Print message only from rank 0 process"""
        if self.is_main_process():
            logger.info(message)


class DistributedRewardAggregator:
    """
    Aggregates rewards across distributed processes for consistent training.
    """
    
    def __init__(self, dist_manager: DistributedTrainingManager):
        self.dist_manager = dist_manager
        
    def aggregate_rewards(
        self,
        local_rewards: torch.Tensor,
        aggregation_method: str = "mean"
    ) -> torch.Tensor:
        """
        Aggregate rewards across all processes.
        
        Args:
            local_rewards: Rewards computed on local batch
            aggregation_method: How to aggregate ('mean', 'sum', 'max', 'consensus')
            
        Returns:
            Aggregated rewards
        """
        if not self.dist_manager.is_distributed:
            return local_rewards
            
        if aggregation_method == "mean":
            return self._aggregate_mean(local_rewards)
        elif aggregation_method == "sum":
            return self._aggregate_sum(local_rewards)
        elif aggregation_method == "max":
            return self._aggregate_max(local_rewards)
        elif aggregation_method == "consensus":
            return self._aggregate_consensus(local_rewards)
        else:
            raise ValueError(f"Unknown aggregation method: {aggregation_method}")
    
    def _aggregate_mean(self, rewards: torch.Tensor) -> torch.Tensor:
        """Average rewards across processes"""
        total_rewards = self.dist_manager.all_reduce_tensor(rewards)
        return total_rewards / self.dist_manager.world_size
    
    def _aggregate_sum(self, rewards: torch.Tensor) -> torch.Tensor:
        """Sum rewards across processes"""
        return self.dist_manager.all_reduce_tensor(rewards)
    
    def _aggregate_max(self, rewards: torch.Tensor) -> torch.Tensor:
        """Take maximum rewards across processes"""
        return self.dist_manager.all_reduce_tensor(rewards, op=dist.ReduceOp.MAX)
    
    def _aggregate_consensus(self, rewards: torch.Tensor) -> torch.Tensor:
        """Use consensus mechanism for reward aggregation"""
        # Gather all rewards
        all_rewards = self.dist_manager.all_gather_tensor(rewards)
        
        # Simple consensus: use median
        stacked_rewards = torch.stack(all_rewards, dim=0)
        consensus_rewards = torch.median(stacked_rewards, dim=0)[0]
        
        return consensus_rewards


def find_free_port() -> int:
    """Find a free port for distributed training initialization"""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(('', 0))
        s.listen(1)
        port = s.getsockname()[1]
    return port


def setup_single_node_distributed(
    rank: int,
    world_size: int,
    fn,
    backend: str = "nccl",
    *args
):
    """
    Setup single-node multi-GPU distributed training.
    
    Args:
        rank: Process rank
        world_size: Total number of processes
        fn: Function to run in distributed mode
        backend: Distributed backend
        *args: Arguments to pass to fn
    """
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = str(find_free_port())
    os.environ['RANK'] = str(rank)
    os.environ['LOCAL_RANK'] = str(rank)
    os.environ['WORLD_SIZE'] = str(world_size)
    
    # Run the function
    fn(rank, *args)


def launch_distributed_training(
    training_fn,
    num_gpus: int,
    *args
):
    """
    Launch distributed training on multiple GPUs.
    
    Args:
        training_fn: Training function to run
        num_gpus: Number of GPUs to use
        *args: Arguments to pass to training function
    """
    if num_gpus <= 1:
        logger.info("Single GPU training")
        training_fn(0, *args)
        return
    
    logger.info(f"Launching distributed training on {num_gpus} GPUs")
    
    mp.spawn(
        setup_single_node_distributed,
        args=(num_gpus, training_fn, "nccl", *args),
        nprocs=num_gpus,
        join=True
    )


class DistributedMetricsAggregator:
    """
    Aggregates training metrics across distributed processes.
    """
    
    def __init__(self, dist_manager: DistributedTrainingManager):
        self.dist_manager = dist_manager
        
    def aggregate_metrics(self, local_metrics: Dict[str, float]) -> Dict[str, float]:
        """
        Aggregate metrics across all processes.
        
        Args:
            local_metrics: Metrics from local process
            
        Returns:
            Aggregated metrics
        """
        if not self.dist_manager.is_distributed:
            return local_metrics
            
        aggregated = {}
        
        for key, value in local_metrics.items():
            # Convert to tensor for aggregation
            tensor_val = torch.tensor(value, dtype=torch.float32)
            
            if torch.cuda.is_available():
                tensor_val = tensor_val.cuda(self.dist_manager.local_rank)
            
            # Aggregate and average
            total_val = self.dist_manager.all_reduce_tensor(tensor_val)
            avg_val = total_val / self.dist_manager.world_size
            
            aggregated[key] = avg_val.item()
            
        return aggregated
    
    def gather_all_metrics(self, local_metrics: Dict[str, float]) -> List[Dict[str, float]]:
        """Gather metrics from all processes to rank 0"""
        if not self.dist_manager.is_distributed:
            return [local_metrics]
            
        # Convert to tensors
        keys = list(local_metrics.keys())
        values = [local_metrics[k] for k in keys]
        
        values_tensor = torch.tensor(values, dtype=torch.float32)
        if torch.cuda.is_available():
            values_tensor = values_tensor.cuda(self.dist_manager.local_rank)
        
        # Gather from all processes
        all_values = self.dist_manager.all_gather_tensor(values_tensor)
        
        # Convert back to dictionaries
        all_metrics = []
        for proc_values in all_values:
            proc_metrics = {k: v.item() for k, v in zip(keys, proc_values)}
            all_metrics.append(proc_metrics)
            
        return all_metrics