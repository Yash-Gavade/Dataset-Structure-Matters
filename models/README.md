# 🤖 Models  

This folder contains references to the fine-tuned models used in the project **“Dataset Structure Matters.”**

---

## 📦 Model Variants  

The following models were trained using **TinyLLaMA with LoRA** on different instruction datasets:

- `tinyllama_alpaca/` → trained on Alpaca dataset  
- `tinyllama_dolly/` → trained on Dolly dataset  
- `tinyllama_oasst1/` → trained on OpenAssistant dataset  

---

## ⚠️ Note on Model Availability  

Due to the **large size of the trained models**, they are not fully hosted in this repository.

👉 The complete models are available at the following link:  

🔗 **[MODEL_LINK_HERE]**  

* Google Drive link :

---

## 🎯 Purpose  

These models are used to:

- Analyze how dataset structure affects model behavior  
- Compare performance across datasets  
- Evaluate instruction-following capabilities  

---

## 🔬 Training Details  

- Base Model: TinyLLaMA  
- Method: LoRA (Low-Rank Adaptation)  
- Training setup: consistent across all datasets  

---

## 📊 Related  

- Evaluation results → `outputs/`  
- Visualizations → `results/`  
- Training pipeline → `src/finetuning/`  

---

## 💡 Tip  

For reproducibility, you can download the models from the link above and place them inside this folder following the same structure.
