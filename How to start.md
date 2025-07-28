# COMP90042 Project 2025 – Group 36



This repository contains the implementation for our automated fact-checking system developed for the Natural Language Processing Project (Semester 1, 2025).

---

## Project Team Members
- Lingxiao Qu 
- Yuqing Gao 
- Binhui Zeng 

---

## File Structure

- `NLP_Project_2025.ipynb`: Main Jupyter Notebook with all code and results
- `eval.py`: Provided evaluation script for scoring retrieval and classification
- `data`: Directory containing the `train-claims.json`, `dev-claims.json`, `test-claims-unlabelled.json`, and `evidence.json` files

---

## Running on Google Colab

To run this notebook in Google Colab:

1. Open the notebook in Colab.
2. Manually upload the following files via the file sidebar:
   - `train-claims.json`
   - `dev-claims.json`
   - `test-claims-unlabelled.json`
   - `evidence.json`
   - `eval.py`
3. Once uploaded, run all cells from top to bottom.

---

## 1. Dataset Processing

### 1.1 Read and transform data
Load all claim and evidence data, and transfer to pandas DataFrames

### 1.2 Extract evidence IDs and texts
Build mappings between IDs and textual evidence

### 1.3 Embed evidences
Encode all evidence texts using Sentence-BERT (SBERT)

---

## 2. Evidence Retrieval

### 2.1 Model Implementations
We implemented and compared three evidence retrieval approaches:

#### 2.1.1 Baseline using SBERT
Use cosine similarity between claim and evidence embeddings, and directly return top-5 evidence

#### 2.1.2 SBERT Reranker with Random Negatives
A binary classifier trained on positive (gold evidence) and randomly sampled negative examples.

#### 2.1.3 SBERT Reranker with Hard Negatives
A stronger reranker trained on hard negatives selected from SBERT top-20 retrieval.

### 2.2 Evaluation and Prediction
Each method is evaluated on the development set and used for test prediction:

- Evaluation using `eval.py` script  
- Output includes F-scores on dev/test for evidence retrieval

---

## 3. Claim Classification

### 3.1 Fine-tuned Transformer Models
Two strategies for evidence aggregation are explored:

#### 3.1.1 Separate Evidence + Voting
Each evidence is independently classified; final label is determined by majority vote.

#### 3.1.2 Combined Evidence
Combines all evidence texts associated with a claim into one input for classification.

Each model is trained and tested with performance reported separately.

### 3.2 In-Context Learning (ICL)
Maximal 5-shot prompting is explored as an alternative classification method.

#### 3.2.1 Model Setup
Using LLMs for 5-shot inference.

#### 3.2.2 Prompt Engineering
Manually designed three types of prompts (Prompt 1, 2, 3).

#### 3.2.3 Evaluation on Dev Set
Each prompt is evaluated using classification accuracy.

#### 3.2.4 Test Inference
All prompts are used to predict test set labels.

---

## Evaluation Summary

We report the following metrics:

- Evidence Retrieval F1-score on dev and test sets  
- Claim Classification Accuracy  
- Harmonic Mean of retrieval and classification performance  

All scores are computed using the `eval.py` script and logic.

---