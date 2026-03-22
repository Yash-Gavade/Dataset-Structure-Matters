# 📊 Dataset Structure Matters  

> 🎓 Term Paper – Machine Learning for Natural Language Understanding (WS 2025/2026)  
> 🏫 Universität Trier | Computational Linguistics & Digital Humanities  

---

## 📖 Title  
**Dataset Structure Matters: A Dataset-Centric Structural Analysis and Empirical Study of Instruction-Tuning Corpora**

---

## ✨ Abstract  

This study investigates how the structural properties of instruction-tuning datasets influence model behavior. Using three widely adopted datasets—Alpaca, Dolly, and OpenAssistant—this work analyzes dataset characteristics such as length, lexical diversity, entropy, redundancy, and semantic clustering. TinyLLaMA models are fine-tuned using LoRA and evaluated through manual scoring and constraint-following tasks. The results demonstrate that dataset construction significantly impacts model performance, with human-generated data producing more structured and higher-quality outputs.

---

## 📌 Overview  

Instruction fine-tuning is a crucial step in aligning large language models (LLMs) with human intent. While prior research primarily focuses on model architectures and optimization strategies, this project adopts a **dataset-centric perspective**, emphasizing the role of dataset structure in shaping model behavior.

This study compares three datasets representing distinct construction paradigms:

- 🧪 Alpaca → Synthetic (template-based)
- 👩‍💻 Dolly → Human-generated
- 💬 OpenAssistant (OASST1) → Conversational

---

## 🎯 Research Questions  

- RQ1: How do instruction datasets differ in structural properties?  
- RQ2: How are these properties related to dataset composition and similarity?  
- RQ3: How do structural differences affect model behavior after fine-tuning?  

---

## 🚀 Key Contributions  

- Structural analysis of instruction datasets  
- Metrics: length, lexical diversity (TTR), entropy, redundancy, clustering  
- Fine-tuning TinyLLaMA using LoRA  
- Manual evaluation + constraint-following tests  
- Dataset → Model behavior relationship  

---

## 🧪 Methodology  

### Data Preprocessing  
- Unified format: instruction → input → output  
- Cleaning and normalization  
- Conversion of multi-turn dialogues into single-turn pairs  

### Structural Analysis  
- Length distributions  
- Lexical diversity (TTR)  
- Shannon entropy  
- Redundancy (TF-IDF similarity)  
- Clustering (K-means)  

### Fine-Tuning  
- Base model: TinyLLaMA  
- Method: LoRA  

### Evaluation  
- Instruction Following  
- Correctness  
- Clarity  
- Completeness  
- Constraint-following tests  

---

## 📊 Results  

### Structural Differences  

| Dataset         | Characteristics              |
|-----------------|------------------------------|
| Alpaca          | Structured, repetitive       |
| Dolly           | Balanced, human-like         |
| OpenAssistant   | Diverse, conversational      |
### Model Performance  

| Model           | Mean Score        |
|-----------------|------------------|
| Alpaca          | 3.01             |
| Dolly           | **3.61 (Best)**  |
| OpenAssistant   | 3.32             |

### Constraint Following  

| Model           | Success Rate     |
|-----------------|-----------------|
| Alpaca          | 70%             |
| Dolly           | **82% (Best)**  |
| OpenAssistant   | 75%             |


#### 🔍 Key Insights  

The results highlight that **dataset structure plays a crucial role in model behavior after fine-tuning**.  

- **Dolly (human-generated)** achieves the best performance across both manual evaluation and constraint-following tasks, indicating that human-authored data provides richer linguistic variability and better structural guidance.  
- **Alpaca (synthetic)** produces more consistent but often shallow outputs, reflecting its template-based generation process.  
- **OpenAssistant (conversational)** shows higher diversity and longer responses, but lacks structural consistency, leading to moderate performance.
Overall, the findings suggest that **dataset construction methodology directly influences model quality, generalization, and formatting ability**, reinforcing the importance of a dataset-centric approach in instruction tuning.
---

## 📂 Project Structure  

src/  
├── preprocessing/  
├── finetuning/  
├── evaluation/  
├── analysis/  

results/  
├── main_figures/  
├── appendix_figures/  

outputs/  
├── figures/  
├── metrics/  
├── logs/  

paper/  
├── Dataset_Structure_Matters.pdf  

---

## ⚙️ How to Run  

pip install -r requirements.txt  
python run_pipeline.py  

Extended:  
python run_extended.py  

---

## 📄 Paper  

See paper/ folder.

---

## ⚠️ Limitations  

- Manual evaluation is subjective  
- Limited constraints  
- Small base model  

---

## 👤 Author  

Yash Gavade  
M.Sc. Natural Language Processing  
Universität Trier  

---

## 👨‍🏫 Supervisors  

- Prof. Dr. Achim Rettinger  
- Raghvi Baloni (PhD Candidate)  

---

## ⭐ Acknowledgment  

Machine Learning for Natural Language Understanding  
Universität Trier  
