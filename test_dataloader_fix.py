"""
Test the dataloader fix for empty dataset issue.
"""

from transformers import AutoTokenizer
from rpt import DataProcessor

# Test the fix
print("Testing DataProcessor with improved quality filtering...")

tokenizer = AutoTokenizer.from_pretrained("gpt2")
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

data_processor = DataProcessor(
    tokenizer=tokenizer,
    max_length=128,
    min_length=5
)

# Test with short texts that might be filtered out
test_texts = [
    "Hi",
    "Hello world",
    "This is short",
    "A slightly longer text example",
    "Machine learning is amazing and powerful"
]

print(f"Input texts: {test_texts}")

try:
    # Create dataset with quality filtering
    dataset = data_processor.create_dataset(
        texts=test_texts,
        filter_quality=True
    )
    
    print(f"Dataset size after filtering: {len(dataset)}")
    
    if len(dataset) > 0:
        # Try creating dataloader
        dataloader = data_processor.create_dataloader(
            dataset,
            batch_size=2,
            shuffle=False
        )
        
        print(f"Dataloader created successfully!")
        
        # Get a batch
        batch = next(iter(dataloader))
        print(f"Batch shape: {batch['input_ids'].shape}")
        print("✅ Test passed!")
    else:
        print("❌ Dataset is still empty after fix")
        
except Exception as e:
    print(f"❌ Error: {e}")