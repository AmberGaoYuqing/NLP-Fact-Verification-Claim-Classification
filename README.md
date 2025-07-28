# NLP Fact-Checking and Claim Classification

This project implements an **automated fact-checking system** designed to verify claims related to climate science. Given a claim, the system retrieves relevant evidence passages from a knowledge source and classifies the claim into one of four categories:
- **SUPPORTS**
- **REFUTES**
- **NOT_ENOUGH_INFO**
- **DISPUTED**

---

## 1. Project Background

With the rise of misinformation regarding climate science, automated fact-checking has become essential. For example:

**Claim**: The Earth’s climate sensitivity is so low that a doubling of atmospheric CO₂ will result in a surface temperature change of 1°C or less.  
**Evidence**:
1. Early studies estimated a 5–6°C temperature increase if CO₂ doubled.  
2. The 1990 IPCC report estimated 1.5–4.5°C, with a "best guess" of 2.5°C.

From the evidence, the claim is **misleading**. This project addresses the challenge of building a system that retrieves relevant evidence and classifies the claim status automatically.

---

## 2. Dataset

We are provided with:
- **train-claims.json** – Labeled training data
- **dev-claims.json** – Development set for validation
- **test-claims-unlabelled.json** – Test set (without labels)
- **evidence.json** – Knowledge source containing evidence passages
- **eval.py** – Evaluation script

Each claim entry includes:
```json
{
  "claim-2967": {
    "claim_text": "[South Australia] has the most expensive electricity in the world.",
    "claim_label": "SUPPORTS",
    "evidences": ["evidence-67732", "evidence-572512"]
  }
}
```

---

## 3. Approach

Our system consists of two main components:

1. **Evidence Retrieval**

   * Uses search and ranking techniques (e.g., BM25 and embedding-based retrieval) to find relevant passages in `evidence.json`.

2. **Claim Classification**

   * Classifies each claim into one of {SUPPORTS, REFUTES, NOT\_ENOUGH\_INFO, DISPUTED} using neural models such as **LSTM**, **GRU**, or **Transformer-based architectures**.
   * Built with PyTorch and HuggingFace Transformers for pre-trained language models.

---

## 4. Evaluation

We use `eval.py` to evaluate our system. It computes:

1. **Evidence Retrieval F-score (F)**
2. **Claim Classification Accuracy (A)**
3. **Harmonic Mean of F and A**

Example baseline output:

```
$ python eval.py --predictions dev-claims-baseline.json --groundtruth dev-claims.json
Evidence Retrieval F-score (F)    = 0.3377
Claim Classification Accuracy (A) = 0.3506
Harmonic Mean of F and A          = 0.3440
```

---

## 5. Report

The project report (see `final_report.pdf`) includes:

* **System Design**: Architecture of retrieval and classification components.
* **Experimental Results**: Performance compared with baselines.
* **Error Analysis**: Strengths and weaknesses of the model.

---

## 6. Technologies Used

* **Languages**: Python
* **Frameworks**: PyTorch, HuggingFace Transformers
* **Tools**: Google Colab, NumPy, NLTK
* **Evaluation**: Custom evaluation script (F-score, accuracy, harmonic mean)


---

## 7. Results

Our final system achieved:

### 🧠 Evidence Retrieval

| Method                                   | Dev F1     | Test F1    |
| ---------------------------------------- | ---------- | ---------- |
| Baseline (SBERT Top-5)                   | 0.1472     | 0.1420     |
| Reranker w/ Random Negatives             | 0.1580     | 0.1271     |
| **Reranker w/ Hard Negatives (lr=2e-5)** | 0.1652     | **0.1568** |
| Reranker w/ Hard Negatives (lr=3e-5)     | **0.1818** | 0.1473     |

* Final model uses **hard negative reranking** with learning rate = `2e-5`.

### 📊 Claim Classification

| Model                                   | Dev Accuracy | Test Accuracy |
| --------------------------------------- | ------------ | ------------- |
| BERT Classifier (Separate Evidence)     | 0.4675       | 0.4026        |
| **BERT Classifier (Combined Evidence)** | 0.4675       | **0.4675**    |
| ICL – Prompt 3 (QA Format)              | 0.4675       | 0.4026        |

---

## 8. Key Findings

* **Hard negative sampling** improves retrieval by exposing the reranker to semantically similar but incorrect evidence.
* **Concatenating evidence** leads to better classification accuracy than evaluating each evidence separately.
* **Transformer-based classification** is more robust than in-context learning (ICL) with DistilGPT2 in low-resource settings.
* **Prompt engineering** significantly affects ICL performance, but its brittleness limits reliability.

---
## 9. Limitations & Future Work

* Retrieval suffers when gold evidence is lexically distant from the claim.
* Long evidence passages risk truncation under model input limits.
* ICL's performance varies greatly with prompt phrasing and lacks consistency.
* Future improvements may include:

  * Text normalization for recall
  * Chunk-based evidence aggregation
  * Noisy evidence-aware training and prompting strategies
---

## 10. Contributors

Yuqing Gao – Data Processing, Retrieval
Lingxiao Qu – Data Processing, Retrieval
Binhui Zeng – Data Processing, Classification
