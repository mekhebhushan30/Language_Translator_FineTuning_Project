from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

# Load trained model
model_path = "./mt5_model"

tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModelForSeq2SeqLM.from_pretrained(model_path)

# Test sentence
text = "translate English to Dutch: I love programming"

# Tokenize
inputs = tokenizer(text, return_tensors="pt")

# Generate output
outputs = model.generate(**inputs, max_length=50, num_beams=4)

# Decode
result = tokenizer.decode(outputs[0], skip_special_tokens=True)

print("Translation:", result)