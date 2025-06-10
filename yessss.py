import torch
from transformers import AutoTokenizer, AutoModel
from rpt import RPTTrainer, RPTModel, RewardSystem, DataProcessor

# Load a pre-trained model and tokenizer
tokenizer = AutoTokenizer.from_pretrained("gpt2")
base_model = AutoModel.from_pretrained("gpt2")

# Create RPT model with value head
rpt_model = RPTModel(
    base_model=base_model,
    tokenizer=tokenizer,
    add_value_head=True
)

# Setup reward system
reward_system = RewardSystem(
    reward_type="hybrid",  # Combines accuracy and confidence
    reward_scale=1.0
)

# Prepare your data
data_processor = DataProcessor(tokenizer=tokenizer)
texts = ["Your training texts here...", "Another example..."]
dataset = data_processor.create_dataset(texts, split_ratio=0.9)
train_dataset, val_dataset = dataset

# Create data loaders
train_loader = data_processor.create_dataloader(
    train_dataset, 
    batch_size=8, 
    shuffle=True
)
val_loader = data_processor.create_dataloader(
    val_dataset, 
    batch_size=8, 
    shuffle=False
)

# Setup optimizer
optimizer = torch.optim.AdamW(rpt_model.parameters(), lr=5e-5)

# Create trainer
trainer = RPTTrainer(
    model=rpt_model,
    reward_system=reward_system,
    optimizer=optimizer,
    train_dataloader=train_loader,
    val_dataloader=val_loader,
    max_epochs=3,
    output_dir="./rpt_output"
)

# Start training
results = trainer.train()
