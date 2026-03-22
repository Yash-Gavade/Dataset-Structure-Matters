# 🧠 Source Code Overview  

This directory contains the core implementation of the project **“Dataset Structure Matters.”**  
It is organized into modular components representing each stage of the experimental pipeline.

---

## 📁 Structure  

```
src/
├── preprocessing/
├── finetuning/
├── evaluation/
├── analysis/
```

---

## 🔹 preprocessing/

Handles dataset preparation and standardization.

Includes:
- Data cleaning and normalization  
- Format unification (`instruction → input → output`)  
- Conversion of multi-turn conversations into single-turn samples  

---

## 🔹 finetuning/

Contains model training logic.

Includes:
- Fine-tuning TinyLLaMA using LoRA  
- Training configuration and setup  
- Model saving and checkpoint handling  

---

## 🔹 evaluation/

Responsible for evaluating model outputs.

Includes:
- Prompt-based evaluation  
- Manual scoring pipeline  
- Constraint-following evaluation  
- Comparison of model outputs  

---

## 🔹 analysis/

Handles data analysis and visualization.

Includes:
- Structural analysis (length, entropy, clustering)  
- Metric computation  
- Plot generation for figures  
- Summary statistics  

---

## 🔄 Workflow  

The modules work together in the following pipeline:

1. **Preprocessing** → Prepare and standardize datasets  
2. **Finetuning** → Train models on each dataset  
3. **Evaluation** → Assess model outputs  
4. **Analysis** → Generate insights and visualizations  

---

## 🎯 Purpose  

This modular structure ensures:

- Clear separation of responsibilities  
- Easy debugging and extension  
- Reproducibility of experiments  
- Scalability for future research  

---

## ⚙️ Integration  

These modules are executed through:

```
scripts/run_pipeline.py
scripts/run_extended.py
```

---

## ⚠️ Notes  

- Each module is independent but connected via the pipeline  
- Outputs are stored in the `outputs/` directory  
- Final visualizations are stored in `results/`  

---

## 🔬 Reproducibility  

- Same experimental setup across datasets  
- Consistent evaluation prompts  
- Controlled training configuration (TinyLLaMA + LoRA)  

---

## 📄 Related  

- Data: `data/`  
- Outputs: `outputs/`  
- Results: `results/`  
- Paper: `paper/`  
