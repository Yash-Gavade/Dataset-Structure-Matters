# ⚙️ Scripts Overview  

This folder contains executable scripts used to run the complete experimental pipeline for the project **“Dataset Structure Matters.”**

These scripts provide a simple interface to reproduce the full workflow, from data preprocessing to evaluation and analysis.

---

## 📁 File Structure  

scripts/
├── run_pipeline.py
├── run_extended.py

---

## 🚀 Available Scripts  

### 🔹 run_pipeline.py  

This script runs the **core experimental pipeline**, including:

- Data preprocessing  
- Structural analysis  
- Model fine-tuning (TinyLLaMA + LoRA)  
- Evaluation (manual + constraint-based)  

👉 Use this for standard reproducibility.

---

### 🔹 run_extended.py  

This script runs an **extended version of the pipeline**, including:

- Additional structural metrics  
- Advanced dataset analysis  
- Extended evaluation outputs  
- Additional logging and statistics  

👉 Use this for deeper analysis and extended results.

---

## ⚙️ Usage  

Make sure dependencies are installed:

pip install -r requirements.txt  

### Run standard pipeline:
python scripts/run_pipeline.py  

### Run extended pipeline:
python scripts/run_extended.py  

---

## 🔄 Workflow Summary  

The scripts internally execute the following steps:

1. Dataset loading and preprocessing  
2. Structural analysis (length, entropy, clustering, etc.)  
3. Model fine-tuning using TinyLLaMA + LoRA  
4. Evaluation using prompts and scoring  
5. Result generation (metrics, figures, logs)  

---

## 📦 Output  

Running these scripts will generate outputs in:

- outputs/figures/  
- outputs/metrics/  
- outputs/logs/  

---

## ⚠️ Notes  

- Ensure datasets are available in the `data/` directory  
- Training may take time depending on hardware  
- GPU is recommended for fine-tuning  

---

## 🎯 Purpose  

These scripts ensure:

- Full reproducibility of experiments  
- Easy execution of the research pipeline  
- Clear separation between core and extended analysis  

