# SMS Messages Classifier (Naive Bayes)

A practical binary text-classification project that detects spam SMS messages using a Naive Bayes model.

## Problem Statement

Unwanted promotional and fraudulent SMS messages are common in real-world communication. Manually filtering spam is slow and inconsistent, so this project focuses on automatically classifying SMS messages into spam or ham. The goal is to learn patterns in message text and word usage from labeled examples, then generalize well to unseen messages.

## Project Overview

This project trains a supervised machine learning classifier on the SMS Spam Collection dataset. It is a focused baseline implementation that uses a standard text-classification pipeline from preprocessing to evaluation.

- Task type: Supervised learning
- Problem type: Binary classification
- Labels: ham (0), spam (1)
- Model: Multinomial Naive Bayes
- Text representation: Bag of Words using CountVectorizer

The workflow is implemented in the notebook:
- Naive_bayes.ipynb

**For deeper technical details (step-by-step methodology, code, outputs, and analysis), open and review Naive_bayes.ipynb.**

## Dataset

- File: SMSSpamCollection
- Original source: UCI SMS Spam Collection
- Size: (5572 * 2)

## Repository Structure

- Naive_bayes.ipynb: Full end-to-end implementation and evaluation
- SMSSpamCollection: Dataset file used for training/testing
- assets/: Images used in notebook explanations
  - sms_dataset_demo.png
  - term_by_doc_matrix.png

## How To Run

1. Create/activate a Python environment (such as venv).
2. Install dependencies:

```bash
pip install pandas scikit-learn
```

3. Open and run all cells in:

- Naive_bayes.ipynb

## Acknowledgement

This project was completed with support from the Udacity Supervised Learning Course. The notebook flow was practiced, then refined into this focused SMS spam classification project.

**Author:** Abdullah Alshammari  
**Email:** abdullah_tecch@gmail.com
