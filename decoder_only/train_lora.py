from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM
from transformers import Trainer, TrainingArguments, DataCollatorForLanguageModeling
from peft import LoraConfig, get_peft_model


# Load dataset
dataset = load_dataset("opus_books", "en-nl", split="train[:2000]")

# Load tokenizer & model
model_name = "gpt2"

tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name)

tokenizer.pad_token = tokenizer.eos_token

# Format data
def format_data(example):
    return {
        "text": f"English: {example['translation']['en']}\nDutch: {example['translation']['nl']}"
    }

dataset = dataset.map(format_data)


# Tokenization + MASKING
def tokenize(example):
    tokenized = tokenizer(
        example["text"],
        max_length=128,
        truncation=True,
        padding="max_length"
    )

    labels = tokenized["input_ids"].copy()

    text = example["text"]
    split_index = text.find("Dutch:")

    if split_index != -1:
        prefix_text = text[:split_index]

        prefix_tokens = tokenizer(
            prefix_text,
            max_length=128,
            truncation=True,
            padding="max_length"
        )

        prefix_len = sum([1 for i in prefix_tokens["input_ids"] if i != tokenizer.pad_token_id])

        labels[:prefix_len] = [-100] * prefix_len

    tokenized["labels"] = labels

    return tokenized

dataset = dataset.map(tokenize, remove_columns=dataset.column_names)

# LoRA configurations
lora_config = LoraConfig(
    r=8,
    lora_alpha=16,
    target_modules=["c_attn"],
    lora_dropout=0.1,
    bias="none",
    task_type="CAUSAL_LM"
)

model = get_peft_model(model, lora_config)

model.print_trainable_parameters()

# Data collator
data_collator = DataCollatorForLanguageModeling(
    tokenizer=tokenizer,
    mlm=False
)

# Training arguments
training_args = TrainingArguments(
    output_dir="./gpt2_lora",
    per_device_train_batch_size=2,
    num_train_epochs=1,
    logging_steps=50,
    save_steps=500,
    save_total_limit=2,
    report_to="none"
)

# Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=dataset,
    data_collator=data_collator
)

# Train
trainer.train()

# Save
model.save_pretrained("./gpt2_lora")
tokenizer.save_pretrained("./gpt2_lora")