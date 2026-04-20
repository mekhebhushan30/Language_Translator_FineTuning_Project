from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from transformers import Trainer, TrainingArguments
from preprocess import preprocess

# Load dataset
dataset = load_dataset("opus_books", "en-nl", split="train[:2000]")

# Load tokenizer & model
model_name = "google/mt5-small"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSeq2SeqLM.from_pretrained(model_name)

# Apply preprocessing
dataset = dataset.map(preprocess)

# Training arguments
training_args = TrainingArguments(
    output_dir="./mt5_model",
    per_device_train_batch_size=4,
    num_train_epochs=1,
    logging_steps=100,
    save_steps=500,
    save_total_limit=2
)

# Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=dataset
)

# Train model
trainer.train()

# Save model
model.save_pretrained("./mt5_model")
tokenizer.save_pretrained("./mt5_model")