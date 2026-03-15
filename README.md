# SMS Messages Classifier (Naive Bayes)

A practical binary text-classification project that detects spam SMS messages using a Multinomial Naive Bayes model.

## Project Overview

This project trains a supervised machine learning classifier on the SMS Spam Collection dataset.

- Task type: Supervised learning
- Problem type: Binary classification
- Labels: ham (0), spam (1)
- Model: Multinomial Naive Bayes
- Text representation: Bag of Words using CountVectorizer

The workflow is implemented in the notebook:
- Naive_bayes.ipynb

## Dataset

- File: SMSSpamCollection
- Original source: UCI SMS Spam Collection

The notebook loads this dataset with two columns:
- label
- sms_message

## Repository Structure

- Naive_bayes.ipynb: Full end-to-end implementation and evaluation
- SMSSpamCollection: Dataset file used for training/testing
- assets/: Images used in notebook explanations
  - sms_dataset_demo.png
  - term_by_doc_matrix.png

## How To Run

1. Create/activate a Python environment.
2. Install dependencies:

```bash
pip install pandas scikit-learn
```

3. Open and run all cells in:

- Naive_bayes.ipynb

## Evaluation In Notebook

The notebook reports standard test-set metrics:
- Accuracy
- Precision
- Recall
- F1 Score

It also includes a synthetic message test section with:
- expected label per message
- predicted label
- success indicator per row
- ham/spam probabilities in percentage format
- synthetic test accuracy

## Notes

- This is a baseline, interpretable text-classification pipeline.
- You can improve performance by testing different vectorization options, adding text preprocessing, or comparing additional models.
