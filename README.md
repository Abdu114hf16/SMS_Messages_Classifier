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

## Author

**Abdullah Alshammari** 

[![Portfolio](https://img.shields.io/badge/Portfolio-alshammari.dev-blue?style=flat&logo=googlechrome&logoColor=white)](https://alshammari.dev)
[![LinkedIn](https://img.shields.io/badge/LinkedIn-Abdullah%20Alshammari-0077B5?style=flat&logo=linkedin&logoColor=white)](https://www.linkedin.com/in/your-linkedin-profile-username)

## This is phase 1 of 2

Phase 2 compares this baseline against random forest, bagging and boosting on
the same features and the same test split:
<https://github.com/Abdu114hf16/Ensemble_SMS_Classifier>

The two phases are presented as one study on the portfolio, because the
comparison is the interesting part.

## Limitations

- 5,572 messages, collected some time ago. **Spam language adapts**, so these
  numbers do not describe current traffic.
- The corpus is imbalanced, so accuracy alone is misleading: a model that never
  predicts spam already scores above 80% and is useless.
- A random split can overestimate robustness when a corpus contains
  near-duplicate messages, which SMS spam corpora commonly do.
- Bag-of-words discards word order, so phrase-level patterns are invisible.
- A production filter needs monitoring for drift, adversarial rewording,
  multilingual input and sender reputation. None of that is covered here.

## Case study

A full write-up: the business question, the method, the evidence, and what the
result does not support.

<https://alshammari.dev/projects/sms-spam-model-comparison/>

## License

MIT. See [LICENSE](LICENSE).
