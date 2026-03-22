
# 📂 Dataset Description  

This folder contains the datasets used in the study **“Dataset Structure Matters”**, including raw datasets and lightweight preview samples for quick inspection.

---

## 📊 Datasets Included  

The following instruction-tuning datasets are used:

- 🧪 Alpaca (Synthetic dataset)
- 👩‍💻 Dolly (Human-generated dataset)
- 💬 OpenAssistant (OASST1) (Conversational dataset)

---

## 📁 File Structure  

data/
├── alpaca.zip
├── dolly.zip
├── oasst1.zip
├── alpaca_preview.jsonl
├── dolly_preview.jsonl
├── oasst1_preview.jsonl

---

## 📦 Dataset Files  

### Full Datasets (.zip)
- Contain the complete raw datasets used for preprocessing and fine-tuning  
- May be large in size  
- Used in the full experimental pipeline  

### Preview Files (.jsonl)
- Small subsets of the datasets  
- Useful for quick inspection and understanding format  
- Each line follows:

{
  "instruction": "...",
  "input": "...",
  "output": "..."
}

---

## 🔄 Preprocessing  

All datasets are standardized into a unified format:

instruction → input → output

Steps include:
- Cleaning and normalization  
- Removing noise and inconsistencies  
- Converting multi-turn dialogues into single-turn instruction-response pairs  

---

## 🎯 Purpose  

These datasets are used to analyze how different dataset construction methods affect:

- Structural properties  
- Linguistic diversity  
- Model behavior after fine-tuning  

---

## ⚠️ Notes  

- Original datasets are publicly available  
- This repository provides processed versions for reproducibility  
- Large files are compressed for GitHub compatibility  

---

## 🔗 References  

See the main paper for full citations and dataset details.
