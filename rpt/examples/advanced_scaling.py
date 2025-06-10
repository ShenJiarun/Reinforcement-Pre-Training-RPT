"""
Advanced RPT Scaling Example

This script demonstrates advanced scaling techniques for RPT training,
including distributed training, memory optimization, and large-scale data processing.
"""

import torch
import torch.distributed as dist
import logging
import os
import argparse
from transformers import AutoTokenizer, AutoModel
from torch.optim import AdamW
from torch.optim.lr_scheduler import OneCycleLR

# Import RPT components
from rpt import (
    RPTModel, 
    RPTTrainer, 
    RewardSystem, 
    AdaptiveRewardSystem,
    DataProcessor,
    ScalingUtils,
    ScalingConfig,
    RPTMetrics
)

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Advanced RPT Scaling Example")
    
    # Model arguments
    parser.add_argument("--model_name", type=str, default="gpt2-medium", 
                       help="HuggingFace model name")
    parser.add_argument("--max_length", type=int, default=1024,
                       help="Maximum sequence length")
    
    # Training arguments
    parser.add_argument("--batch_size", type=int, default=16,
                       help="Batch size per device")
    parser.add_argument("--learning_rate", type=float, default=1e-4,
                       help="Learning rate")
    parser.add_argument("--num_epochs", type=int, default=5,
                       help="Number of training epochs")
    parser.add_argument("--gradient_accumulation_steps", type=int, default=4,
                       help="Gradient accumulation steps")
    
    # Scaling arguments
    parser.add_argument("--distributed", action="store_true",
                       help="Enable distributed training")
    parser.add_argument("--gradient_checkpointing", action="store_true",
                       help="Enable gradient checkpointing")
    parser.add_argument("--auto_batch_size", action="store_true",
                       help="Automatically find optimal batch size")
    parser.add_argument("--compile_model", action="store_true",
                       help="Compile model with torch.compile")
    
    # Data arguments
    parser.add_argument("--data_path", type=str, required=True,
                       help="Path to training data")
    parser.add_argument("--data_format", type=str, default="auto",
                       choices=["txt", "json", "jsonl", "auto"],
                       help="Data format")
    
    # Output arguments
    parser.add_argument("--output_dir", type=str, default="./rpt_advanced_output",
                       help="Output directory")
    parser.add_argument("--save_steps", type=int, default=2000,
                       help="Save checkpoint every N steps")
    parser.add_argument("--eval_steps", type=int, default=1000,
                       help="Evaluate every N steps")
    parser.add_argument("--logging_steps", type=int, default=50,
                       help="Log every N steps")
    
    return parser.parse_args()


def setup_distributed():
    """Setup distributed training."""
    if "RANK" in os.environ and "WORLD_SIZE" in os.environ:
        rank = int(os.environ["RANK"])
        world_size = int(os.environ["WORLD_SIZE"])
        local_rank = int(os.environ.get("LOCAL_RANK", 0))
        
        dist.init_process_group("nccl", rank=rank, world_size=world_size)
        torch.cuda.set_device(local_rank)
        
        return rank, world_size, local_rank
    else:
        return 0, 1, 0


def main():
    """Main training function."""
    args = parse_args()
    
    # Setup distributed training if requested
    rank, world_size, local_rank = 0, 1, 0
    if args.distributed:
        rank, world_size, local_rank = setup_distributed()
    
    # Only log from main process
    if rank == 0:
        logger.info("Starting Advanced RPT Scaling Example")
        logger.info(f"Arguments: {args}")
        logger.info(f"World size: {world_size}, Rank: {rank}")
    
    # Device setup
    if torch.cuda.is_available():
        device = torch.device(f"cuda:{local_rank}")
        torch.cuda.set_device(device)
    else:
        device = torch.device("cpu")
    
    if rank == 0:
        logger.info(f"Using device: {device}")
    
    # Setup scaling configuration
    scaling_config = ScalingConfig(
        use_distributed=args.distributed,
        world_size=world_size,
        rank=rank,
        local_rank=local_rank,
        gradient_checkpointing=args.gradient_checkpointing,
        compile_model=args.compile_model,
        auto_batch_size=args.auto_batch_size,
        mixed_precision=True,
        max_batch_size=args.batch_size * 4,  # Allow search up to 4x base batch size
        min_batch_size=max(1, args.batch_size // 4)
    )
    
    scaling_utils = ScalingUtils(scaling_config)
    
    # Get system information
    if rank == 0:
        system_info = scaling_utils.get_system_info()
        logger.info(f"System info: {system_info}")
    
    # Load model and tokenizer
    if rank == 0:
        logger.info(f"Loading model: {args.model_name}")
    
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    base_model = AutoModel.from_pretrained(args.model_name)
    
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # Create RPT model
    rpt_model = RPTModel(
        base_model=base_model,
        tokenizer=tokenizer,
        add_value_head=True,
        device=device
    )
    
    # Apply memory optimizations
    rpt_model = scaling_utils.optimize_model_memory(rpt_model)
    
    # Wrap for distributed training
    if args.distributed:
        rpt_model = scaling_utils.wrap_model_for_distributed(rpt_model)
    
    # Get memory estimates
    if rank == 0:
        memory_estimates = scaling_utils.estimate_memory_usage(
            model=rpt_model,
            batch_size=args.batch_size,
            sequence_length=args.max_length
        )
        logger.info(f"Memory estimates: {memory_estimates}")
    
    # Setup adaptive reward system
    reward_system = AdaptiveRewardSystem(
        initial_reward_scale=1.0,
        adaptive_rate=0.01,
        target_positive_ratio=0.6,
        reward_type="hybrid"
    )
    
    # Load and process data
    if rank == 0:
        logger.info(f"Loading data from {args.data_path}")
    
    data_processor = DataProcessor(
        tokenizer=tokenizer,
        max_length=args.max_length,
        reasoning_augmentation=True
    )
    
    # Load texts (only on main process for now - in production, you'd want distributed data loading)
    if rank == 0:
        texts = data_processor.load_text_data(args.data_path, args.data_format)
        logger.info(f"Loaded {len(texts)} texts")
        
        # Create datasets
        train_dataset, val_dataset = data_processor.create_dataset(
            texts=texts,
            split_ratio=0.95,  # Use more data for training in large-scale setting
            shuffle=True,
            filter_quality=True
        )
        
        logger.info(f"Training samples: {len(train_dataset)}")
        logger.info(f"Validation samples: {len(val_dataset)}")
        
        # Get dataset statistics
        train_stats = data_processor.get_data_statistics(train_dataset)
        logger.info(f"Training data stats: {train_stats}")
    else:
        # For distributed training, you'd need to share data across processes
        # This is simplified for the example
        train_dataset = val_dataset = None
    
    # Determine optimal batch size
    optimal_batch_size = args.batch_size
    if args.auto_batch_size and rank == 0:
        sample_input = {
            "input_ids": torch.randint(0, tokenizer.vocab_size, (1, args.max_length)).to(device),
            "attention_mask": torch.ones(1, args.max_length).to(device)
        }
        
        optimal_batch_size = scaling_utils.get_optimal_batch_size(
            model=rpt_model,
            sample_input=sample_input,
            device=device
        )
        
        logger.info(f"Using optimal batch size: {optimal_batch_size}")
    
    # Create data loaders
    if train_dataset is not None:
        train_loader = data_processor.create_dataloader(
            train_dataset,
            batch_size=optimal_batch_size,
            shuffle=True,
            distributed=args.distributed,
            num_workers=4
        )
        
        val_loader = data_processor.create_dataloader(
            val_dataset,
            batch_size=optimal_batch_size,
            shuffle=False,
            distributed=args.distributed,
            num_workers=4
        )
    else:
        train_loader = val_loader = None
    
    # Setup optimizer with proper learning rate scaling for distributed training
    lr = args.learning_rate
    if args.distributed:
        lr = args.learning_rate * world_size  # Scale learning rate
    
    optimizer = AdamW(
        rpt_model.parameters(),
        lr=lr,
        weight_decay=0.01,
        betas=(0.9, 0.95),
        eps=1e-8
    )
    
    # Setup scheduler
    if train_loader is not None:
        total_steps = len(train_loader) * args.num_epochs // args.gradient_accumulation_steps
        scheduler = OneCycleLR(
            optimizer,
            max_lr=lr,
            total_steps=total_steps,
            pct_start=0.1,
            anneal_strategy='cos'
        )
    else:
        scheduler = None
    
    # Create trainer
    if rank == 0:
        logger.info("Creating advanced RPT trainer")
    
    trainer = RPTTrainer(
        model=rpt_model,
        reward_system=reward_system,
        optimizer=optimizer,
        train_dataloader=train_loader,
        val_dataloader=val_loader,
        max_epochs=args.num_epochs,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        max_grad_norm=1.0,
        logging_steps=args.logging_steps,
        eval_steps=args.eval_steps,
        save_steps=args.save_steps,
        output_dir=args.output_dir,
        mixed_precision=True,
        device=device
    )
    
    # Setup metrics (only on main process)
    metrics = None
    if rank == 0:
        metrics = RPTMetrics(track_detailed_stats=True)
    
    # Get scaling recommendations
    if rank == 0:
        recommendations = scaling_utils.get_scaling_recommendations(
            model=rpt_model,
            target_batch_size=optimal_batch_size,
            sequence_length=args.max_length
        )
        logger.info(f"Scaling recommendations: {recommendations}")
    
    # Start training
    if rank == 0:
        logger.info("Starting advanced RPT training...")
    
    try:
        results = trainer.train()
        
        if rank == 0:
            logger.info("Training completed successfully!")
            logger.info(f"Final results: {results}")
            
            # Save final metrics and plots
            if metrics:
                metrics.plot_metrics(
                    save_path=f"{args.output_dir}/advanced_training_metrics.png"
                )
                metrics.export_metrics(f"{args.output_dir}/advanced_metrics.json")
            
            # Save scaling info
            scaling_info = {
                "scaling_config": scaling_config.__dict__,
                "system_info": system_info,
                "memory_estimates": memory_estimates,
                "recommendations": recommendations,
                "optimal_batch_size": optimal_batch_size,
                "world_size": world_size
            }
            
            import json
            with open(f"{args.output_dir}/scaling_info.json", 'w') as f:
                json.dump(scaling_info, f, indent=2, default=str)
        
    except Exception as e:
        if rank == 0:
            logger.error(f"Training failed: {e}")
        raise
    finally:
        # Cleanup distributed training
        if args.distributed:
            scaling_utils.cleanup_distributed()
    
    if rank == 0:
        logger.info("Advanced RPT scaling example completed!")


if __name__ == "__main__":
    main()