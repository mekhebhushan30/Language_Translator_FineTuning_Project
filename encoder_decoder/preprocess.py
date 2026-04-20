from datasets import load_dataset
from transformers import AutoTokenizer

# Load dataset
dataset = load_dataset("opus_books", "en-nl", split="train[:2000]")

# Load tokenizer
tokenizer = AutoTokenizer.from_pretrained("google/mt5-small")

# Cleaning function
def clean_text(text):
    if text is None:
        return None
    if "Source:" in text:
        return None
    if len(text.strip()) < 5:
        return None
    return text

# Preprocessing function
def preprocess(example):
    en_text = clean_text(example["translation"]["en"])
    nl_text = clean_text(example["translation"]["nl"])

    # Skip bad rows
    if en_text is None or nl_text is None:
        return None

    input_text = "translate English to Dutch: " + en_text
    target_text = nl_text

    # Tokenize input
    model_input = tokenizer(
        input_text,
        max_length=128,
        truncation=True,
        padding="max_length"
    )

    # Tokenize output
    labels = tokenizer(
        target_text,
        max_length=128,
        truncation=True,
        padding="max_length"
    )

    model_input["labels"] = labels["input_ids"]

    return model_input

# Apply preprocessing
dataset = dataset.map(preprocess)

# Remove empty rows
dataset = dataset.filter(lambda x: len(x["input_ids"]) > 0)

# Check output
print(dataset[0])
print("Total samples:", len(dataset))