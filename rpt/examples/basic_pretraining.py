"""
Basic RPT Pre-training Example

This script demonstrates how to use the RPT package for basic
reinforcement pre-training of language models.
"""

import torch
import logging
from transformers import AutoTokenizer, AutoModel
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR

# Import RPT components
from rpt import (
    RPTModel, 
    RPTTrainer, 
    RewardSystem, 
    DataProcessor,
    RPTMetrics
)

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def main():
    """Main training function."""
    
    # Configuration
    CONFIG = {
        "model_name": "gpt2",  # Can be any HuggingFace model
        "max_length": 512,
        "batch_size": 4,
        "learning_rate": 5e-5,
        "num_epochs": 3,
        "output_dir": "./rpt_basic_output",
        "save_steps": 1000,
        "eval_steps": 500,
        "logging_steps": 100,
    }
    
    logger.info("Starting Basic RPT Pre-training")
    logger.info(f"Configuration: {CONFIG}")
    
    # Device setup
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")
    
    # Load tokenizer and base model
    logger.info(f"Loading model: {CONFIG['model_name']}")
    tokenizer = AutoTokenizer.from_pretrained(CONFIG["model_name"])
    base_model = AutoModel.from_pretrained(CONFIG["model_name"])
    
    # Ensure pad token exists
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # Create RPT model
    logger.info("Creating RPT model with value head")
    rpt_model = RPTModel(
        base_model=base_model,
        tokenizer=tokenizer,
        add_value_head=True,
        device=device
    )
    
    # Setup reward system
    logger.info("Setting up reward system")
    reward_system = RewardSystem(
        reward_type="hybrid",  # Combines accuracy and confidence
        reward_scale=1.0,
        confidence_threshold=0.5
    )
    
    # Prepare sample data (replace with your own dataset)
    sample_texts = [
        "The quick brown fox jumps over the lazy dog. This is a classic example sentence used in typography.",
        "Machine learning is a subset of artificial intelligence that enables computers to learn without being explicitly programmed.",
        "The human brain contains approximately 86 billion neurons, each connected to thousands of others.",
        "Climate change refers to long-term shifts in global temperatures and weather patterns caused by human activities.",
        "Python is a high-level programming language known for its simplicity and readability.",
        "The theory of relativity, developed by Albert Einstein, revolutionized our understanding of space and time.",
        "Photosynthesis is the process by which plants convert sunlight, carbon dioxide, and water into glucose and oxygen.",
        "The internet has transformed how we communicate, work, and access information in the modern world.",
        "DNA contains the genetic instructions necessary for the development and functioning of all living organisms.",
        "Renewable energy sources like solar and wind power are crucial for sustainable development.",
    ] * 10  # Repeat for more training data
    
    # Process data
    logger.info("Processing training data")
    data_processor = DataProcessor(
        tokenizer=tokenizer,
        max_length=CONFIG["max_length"],
        reasoning_augmentation=True
    )
    
    # Create train/validation split
    train_dataset, val_dataset = data_processor.create_dataset(
        texts=sample_texts,
        split_ratio=0.8,
        shuffle=True,
        filter_quality=True
    )
    
    logger.info(f"Training samples: {len(train_dataset)}")
    logger.info(f"Validation samples: {len(val_dataset)}")
    
    # Create data loaders
    train_loader = data_processor.create_dataloader(
        train_dataset,
        batch_size=CONFIG["batch_size"],
        shuffle=True
    )
    
    val_loader = data_processor.create_dataloader(
        val_dataset,
        batch_size=CONFIG["batch_size"],
        shuffle=False
    )
    
    # Setup optimizer and scheduler
    optimizer = AdamW(
        rpt_model.parameters(),
        lr=CONFIG["learning_rate"],
        weight_decay=0.01
    )
    
    scheduler = CosineAnnealingLR(
        optimizer,
        T_max=len(train_loader) * CONFIG["num_epochs"]
    )
    
    # Create trainer
    logger.info("Creating RPT trainer")
    trainer = RPTTrainer(
        model=rpt_model,
        reward_system=reward_system,
        optimizer=optimizer,
        train_dataloader=train_loader,
        val_dataloader=val_loader,
        max_epochs=CONFIG["num_epochs"],
        gradient_accumulation_steps=1,
        max_grad_norm=1.0,
        logging_steps=CONFIG["logging_steps"],
        eval_steps=CONFIG["eval_steps"],
        save_steps=CONFIG["save_steps"],
        output_dir=CONFIG["output_dir"],
        mixed_precision=True,
        device=device
    )
    
    # Setup metrics tracking
    metrics = RPTMetrics(track_detailed_stats=True)
    
    # Start training
    logger.info("Starting training...")
    try:
        results = trainer.train()
        
        logger.info("Training completed successfully!")
        logger.info(f"Final results: {results}")
        
        # Generate some sample text to test the model
        logger.info("Testing trained model with sample generation...")
        test_generation(rpt_model, tokenizer, device)
        
        # Plot and save metrics
        logger.info("Saving training metrics...")
        trainer_logs = results.get("train_logs", [])
        if trainer_logs:
            # Extract metrics for plotting
            for log_entry in trainer_logs[-100:]:  # Last 100 steps
                if "step" in log_entry:
                    # Create dummy tensors for metrics update
                    dummy_predictions = torch.randn(1, 10, tokenizer.vocab_size)
                    dummy_targets = torch.randint(0, tokenizer.vocab_size, (1, 10))
                    dummy_rewards = torch.randn(1, 10)
                    
                    metrics.update_metrics(
                        step=log_entry["step"],
                        predictions=dummy_predictions,
                        targets=dummy_targets,
                        rewards=dummy_rewards,
                        additional_metrics={
                            "loss": log_entry.get("loss", 0.0),
                            "reward": log_entry.get("reward", 0.0)
                        }
                    )
            
            # Plot metrics
            metrics.plot_metrics(
                save_path=f"{CONFIG['output_dir']}/training_metrics.png"
            )
            
            # Export metrics
            metrics.export_metrics(f"{CONFIG['output_dir']}/metrics.json")
        
    except Exception as e:
        logger.error(f"Training failed: {e}")
        raise
    
    logger.info("Basic RPT training example completed!")


def test_generation(model, tokenizer, device, max_length=50):
    """Test the trained model with sample generation."""
    
    model.eval()
    
    test_prompts = [
        "The future of artificial intelligence",
        "In a world where",
        "Scientists have discovered"
    ]
    
    for prompt in test_prompts:
        logger.info(f"\nTesting prompt: '{prompt}'")
        
        # Tokenize prompt
        inputs = tokenizer(
            prompt,
            return_tensors="pt",
            add_special_tokens=True
        )
        input_ids = inputs.input_ids.to(device)
        
        # Generate with reasoning
        with torch.no_grad():
            generation_result = model.generate_with_reasoning(
                input_ids=input_ids,
                max_length=input_ids.shape[1] + max_length,
                temperature=0.8,
                do_sample=True,
                reasoning_steps=2
            )
        
        # Decode generated text
        generated_ids = generation_result["generated_ids"]
        generated_text = tokenizer.decode(generated_ids[0], skip_special_tokens=True)
        
        logger.info(f"Generated: {generated_text}")
        logger.info(f"Reasoning scores: {generation_result['reasoning_scores'].tolist()}")


if __name__ == "__main__":
    main()