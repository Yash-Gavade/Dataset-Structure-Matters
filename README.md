# 📊 Dataset Structure Matters

## A Dataset-Centric Structural Analysis and Empirical Study of Instruction-Tuning Corpora

---

## 👨‍🎓 Author

**Yash Sakharam Gavade**
M.Sc. Natural Language Processing
Universität Trier

---

## 📌 Abstract

Instruction fine-tuning has become a critical step in aligning large language models (LLMs) with human tasks. While most research focuses on model architectures and training strategies, relatively little attention has been given to the structure of instruction datasets themselves.

This project presents a **dataset-centric study** analyzing how the structural properties of instruction-tuning datasets influence model behavior. Three widely used datasets — **Alpaca**, **Dolly**, and **OpenAssistant (OASST1)** — are analyzed using quantitative metrics such as length distributions, lexical diversity, entropy, redundancy, and semantic clustering.

A controlled experiment is conducted by fine-tuning the **TinyLLaMA** model using **LoRA (Low-Rank Adaptation)** on each dataset. The resulting models are evaluated using manual scoring and constraint-following tests.

The findings demonstrate that dataset construction methodology significantly impacts both dataset structure and model performance. Human-authored datasets produce more balanced and high-quality outputs, while synthetic datasets tend to be more structured but repetitive, and conversational datasets exhibit higher diversity but less consistency.

---

## 🎯 Motivation

Most existing work in NLP focuses on:

* Model architectures (Transformers, LLMs)
* Training strategies (pretraining, RLHF)
* Evaluation benchmarks

However, **datasets are often treated as static inputs**, even though they directly influence:

* Learning patterns
* Output quality
* Generalization behavior

👉 This project explores a **data-centric perspective**, asking:

* How do instruction datasets differ structurally?
* Do these differences affect model behavior?
* Can dataset quality explain performance differences?

---

## 📚 Datasets

### 🔹 Alpaca (Synthetic Dataset)

* Generated using self-instruct method
* High template consistency
* Structured and repetitive

### 🔹 Dolly (Human-Authored Dataset)

* Created by annotators
* Natural language variation
* Balanced structure

### 🔹 OpenAssistant (Conversational Dataset)

* Multi-turn dialogue data
* High diversity and variability
* Longer responses

---

## ⚙️ Methodology

The project follows a **two-stage pipeline**:

### Stage 1: Structural Analysis

* Standardization of datasets
* Metric computation
* Pattern analysis

### Stage 2: Model Evaluation

* Fine-tuning TinyLLaMA
* Generating responses
* Manual + constraint evaluation

---

## 🔄 Complete Workflow

![Workflow](images/fig_4_1_workflow.png)

---

## 🧹 Data Preprocessing

To ensure fair comparison, all datasets are converted into a unified format:

* Instruction
* Input (optional)
* Output

### Steps:

* Remove null / incomplete records
* Normalize text formatting
* Standardize schema
* Convert multi-turn conversations → single-turn pairs

---

## 📊 Structural Analysis

### 1. Length Distribution

Measures instruction and response length.

👉 Insight:

* Synthetic → short & consistent
* Conversational → long & variable

---

### 2. Lexical Diversity (TTR)

[
TTR = \frac{\text{Unique Tokens}}{\text{Total Tokens}}
]

Higher values indicate richer vocabulary.

---

### 3. Entropy

[
H = -\sum p(w)\log p(w)
]

Measures variability in token usage.

---

### 4. Redundancy Analysis

* Exact duplicate detection
* TF-IDF similarity
* Cosine similarity

---

### 5. Semantic Clustering

* K-Means clustering
* Embedding-based grouping

👉 Shows conceptual coverage of dataset

---

## 📈 Key Visualizations

### Instruction Length

![Alpaca](images/hist_instr_len_alpaca.png)
![Dolly](images/hist_instr_len_dolly.png)
![OASST1](images/hist_instr_len_oasst1.png)

---

### Output Length

![Output Alpaca](images/hist_out_len_alpaca.png)
![Output Dolly](images/hist_out_len_dolly.png)
![Output OASST1](images/hist_out_len_oasst1.png)

---

### Vocabulary & Entropy

![Vocab](images/vocab_size_v2.png)
![Entropy](images/output_entropy_v2.png)

---

## 🤖 Model Training

### Model

* **TinyLLaMA**

### Technique

* **LoRA (Low-Rank Adaptation)**

### Why LoRA?

* Efficient training
* Lower memory usage
* Faster experimentation

---

### Training Configuration

| Parameter     | Value |
| ------------- | ----- |
| Learning Rate | 2e-5  |
| Batch Size    | 8     |
| Epochs        | 3     |
| Optimizer     | AdamW |

---

### Training Loss Curve

![Loss](images/training_loss_plot.png)

👉 Loss decreases steadily → stable convergence

---

## 📊 Evaluation

### Evaluation Types

#### 1. Manual Evaluation

Each response scored on:

* Instruction Following
* Correctness
* Clarity
* Completeness

Scale: 1 (Poor) → 4 (Excellent)

---

#### 2. Constraint Evaluation

Tests:

* Bullet points
* Numbered lists
* Length constraints
* Structured format

---

## 📌 Results

### Manual Scores

| Model  | Mean Score |
| ------ | ---------- |
| Alpaca | 3.01       |
| Dolly  | 3.61       |
| OASST1 | 3.32       |

👉 Dolly performs best overall

---

### Key Observations

* **Alpaca**

  * Concise
  * Less detailed
  * Template-like

* **Dolly**

  * Structured
  * Balanced
  * Best performance

* **OASST1**

  * Long responses
  * More conversational
  * Less consistent

---

## 📊 Additional Analysis

### Response Length by Category

![Length](images/category_response_length_v2.png)

### Entropy by Category

![Entropy](images/category_entropy_v2.png)

### Manual Score Heatmap

![Heatmap](images/category_manual_score_heatmap_v2.png)

---

## 🔍 Key Insights

* Dataset structure directly influences model outputs
* Human-authored data → best generalization
* Synthetic data → efficient but repetitive
* Conversational data → diverse but noisy

---

## 📁 Project Structure

```
project/
│
├── data/
├── analysis/
├── finetuning/
├── images/
├── results/
└── README.md
```

---

## 🚀 How to Run

```bash
pip install -r requirements.txt
```

```bash
python analysis/structural_metrics.py
```

```bash
python finetuning/train_model.py
```

```bash
python finetuning/evaluate_model.py
```

```bash
python finetuning/plot_training_loss.py
```

---

## ⚠️ Limitations

* Only 3 datasets analyzed
* Small model (TinyLLaMA)
* Manual evaluation subjectivity

---

## 🔮 Future Work

* Larger models (LLaMA, Mistral)
* More datasets (FLAN, Self-Instruct)
* Automated evaluation
* Advanced semantic analysis

---

## 📌 Conclusion

This project demonstrates that **dataset structure is a critical factor in instruction fine-tuning**. Beyond model architecture, the composition and design of datasets significantly influence how models learn and respond.

Adopting a **data-centric approach** can lead to better model performance, improved alignment, and more reliable outputs.

---

## ⭐ Acknowledgment

Developed as part of
**Machine Learning for Natural Language Understanding**

---

## 📄 License

For academic and research purposes.
