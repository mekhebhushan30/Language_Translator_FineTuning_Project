from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel

# Load base + LoRA
base_model = AutoModelForCausalLM.from_pretrained("gpt2")
model = PeftModel.from_pretrained(base_model, "./gpt2_lora")

tokenizer = AutoTokenizer.from_pretrained("./gpt2_lora")

# Prompt (same as training)
text = "English: I love programming\nDutch:"

inputs = tokenizer(text, return_tensors="pt")

outputs = model.generate(
    **inputs,
    max_length=60,
    temperature=0.7,
    top_p=0.9,
    do_sample=True,
    pad_token_id=tokenizer.eos_token_id
)

result = tokenizer.decode(outputs[0], skip_special_tokens=True)

print(result)