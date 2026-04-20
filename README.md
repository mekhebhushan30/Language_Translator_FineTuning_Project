# Language Translator Fine-Tuning (EN → NL)

## Objective

Build and compare two approaches for English to Dutch translation using transformer models.

---

## Models Used

**1. mT5 (Encoder-Decoder)**

* Used for translation (seq2seq)
* Full fine-tuning

**2. GPT2 + LoRA (Decoder-only)**

* Used LoRA for lightweight fine-tuning
* Faster training, less compute

---

## Dataset

* OPUS Books (EN–NL)
* Used ~2000 samples (CPU limitation)

---

## Approach

This implementation was built step-by-step with debugging and experimentation to understand model behavior.

* Basic cleaning (removed noisy text)
* Tokenization
* Trained:
  * mT5 (full fine-tune)
  * GPT2 (LoRA)
* Evaluated using BLEU score

---

## Results

* **mT5:** Training worked but did not converge well on CPU
* **GPT2 + LoRA:** Faster but struggled with translation

**BLEU Score:** ~0.07

---

## Key Observation

Model performance was low because:

* Training data was general-domain
* Evaluation dataset contained software-specific text

👉 This domain mismatch affected translation quality.

---

## Learnings

* Encoder-decoder models perform better for translation
* Decoder-only models need more data or instruction tuning
* LoRA is useful for low-resource training

---

## Limitations

* CPU training
* Small dataset
* Limited training time

---

## Conclusion

This project successfully implements a full fine-tuning pipeline and compares two model architectures.
While performance is limited, the goal of understanding and implementing the approach is achieved.

---
