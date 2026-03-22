# 📊 Dataset Structure Matters  

### A Dataset-Centric Structural Analysis and Empirical Study of Instruction-Fine-Tuning Corpora  

---

### 🎓 Universität Trier  
**Computational Linguistics and Digital Humanities**  
Department II  

---

### 📘 Term Paper  
**Machine Learning for Natural Language Understanding**  

---

### 👤 Author  
**Yash Gavade**  
Matr.-Nr.: 1757209  
📧 s4yagava@uni-trier.de  

---

### 🎓 Degree  
Master of Science  
Natural Language Processing  

---

### 📅 Semester  
Winter Semester 2025/2026  

---

### 📌 Submission Date  
March 23, 2026  

---

### 👨‍🏫 Supervisors  
- **Prof. Dr. Achim Rettinger**  
- **Raghvi Baloni (PhD Candidate)**  

---
##  Abstract

This study analyzes how instruction dataset structure affects model behavior. Using Alpaca, Dolly, and OpenAssistant, structural properties are examined and TinyLLaMA models are fine-tuned with LoRA. Results show that dataset design significantly impacts performance, with human-generated data producing better outputs.

## 📌 Overview

Instruction fine-tuning is a key step in aligning large language models (LLMs) with human tasks. While most research focuses on model architectures and training strategies, this project investigates a dataset-centric perspective — analyzing how the structure of instruction datasets influences model behavior.

This study analyzes three widely used datasets:
- Alpaca (synthetic)
- Dolly (human-generated)
- OpenAssistant (conversational)

---

## 🎯 Research Questions

- RQ1: How do instruction datasets differ in structural properties?
- RQ2: How are these properties related to dataset composition?
- RQ3: Do structural differences affect model behavior after fine-tuning?

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
- Standardized datasets into instruction → input → output
- Cleaned and normalized data
- Converted multi-turn dialogue to single-turn format

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
- Alpaca: structured, repetitive
- Dolly: balanced, human-like
- OpenAssistant: diverse, conversational

### Model Performance
- Alpaca: 3.01
- Dolly: 3.61 (Best)
- OpenAssistant: 3.32

### Constraint Following
- Alpaca: 70%
- Dolly: 82%
- OpenAssistant: 75%

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

## ⭐ Acknowledgment

Machine Learning for Natural Language Understanding  
Universität Trier  
