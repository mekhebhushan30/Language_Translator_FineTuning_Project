import pandas as pd
import evaluate
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

# Load BLEU
bleu = evaluate.load("sacrebleu")

# Load Excel dataset
df = pd.read_excel("data/Dataset_Challenge_1.xlsx")

# Load model
model_path = "mt5_model"

tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModelForSeq2SeqLM.from_pretrained(model_path)

predictions = []
references = []

for _, row in df.iterrows():
    input_text = "translate English to Dutch: " + str(row["English Source"])

    inputs = tokenizer(input_text, return_tensors="pt")

    outputs = model.generate(
        **inputs,
        max_length=50,
        num_beams=4
    )

    pred = tokenizer.decode(outputs[0], skip_special_tokens=True)
    ref = str(row["Reference Translation"])

    predictions.append(pred)
    references.append([ref])

# BLEU score
result = bleu.compute(predictions=predictions, references=references)

print("BLEU Score (Excel Dataset):", result["score"])

# Sample outputs
print("\nSample Predictions:\n")

for i in range(min(5, len(df))):
    print("EN:", df.iloc[i]["English Source"])
    print("PRED:", predictions[i])
    print("REF:", references[i][0])
    print("-" * 50)