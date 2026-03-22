# 📊 Results Overview  

This folder contains all visualizations and figures generated from the study **“Dataset Structure Matters.”**  
It is divided into main figures used in the paper and additional figures included in the appendix.

---

## 📁 Folder Structure  

results/
├── main_figures/
├── appendix_figures/

---

## 📊 Main Figures  

The `main_figures/` folder contains all key visualizations used in the paper:

- Instruction length distributions  
- Output length distributions  
- Entropy and vocabulary diversity  
- Semantic clustering results  
- Manual evaluation scores  
- Constraint-following performance  

👉 These figures represent the **core findings** of the study.

---

## 📈 Appendix Figures  

The `appendix_figures/` folder contains supplementary visualizations:

- Category-wise response analysis  
- Entropy across task categories  
- Manual scoring heatmaps  
- Training loss plots  
- Additional statistical insights  

👉 These figures provide **extended analysis and deeper insights** beyond the main paper.

---

## 🎯 Purpose  

The results in this folder are used to:

- Compare structural properties across datasets  
- Evaluate model performance after fine-tuning  
- Visualize dataset behavior and model outputs  
- Support findings presented in the research paper  

---

## 📊 Key Insights  

- Human-generated datasets (Dolly) produce the most balanced and high-quality outputs  
- Synthetic datasets (Alpaca) are more structured but repetitive  
- Conversational datasets (OpenAssistant) are diverse but less consistent  
- Dataset structure significantly influences model performance  

---

## ⚠️ Notes  

- Figures are generated using scripts from `src/analysis/`  
- Results are reproducible using the pipeline scripts  
- Some plots are simplified for readability in the paper  

---

## 🔄 Reproducibility  

To regenerate results:

python scripts/run_pipeline.py  

or for extended analysis:

python scripts/run_extended.py  

---

## 📦 Output Source  

Figures are generated from:

- outputs/figures/  
- outputs_extended/  

---

