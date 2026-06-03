# Credit Default Prediction

Predicting serious borrower delinquency using Logistic Regression and Random Forest, with probability calibration and cost-sensitive thresholding aligned to real lending economics.

## Overview

This project builds and evaluates two credit-risk models on the [Give Me Some Credit](https://www.kaggle.com/c/GiveMeSomeCredit) dataset (150,000 borrower records, 11 predictors, 6.7% default rate). The aim is not only to rank borrowers by risk but to produce well-calibrated default probabilities and to choose a decision threshold that reflects the real cost of lending mistakes. In credit lending, failing to flag a defaulter is far more expensive than rejecting a good applicant, and the modelling here is built around that asymmetry.

## Data

* **Source:** Give Me Some Credit (Kaggle), file `cs-training.csv`
* **Size:** 150,000 records, 11 predictors
* **Target:** `SeriousDlqin2yrs`, where 1 indicates serious delinquency within two years
* **Class balance:** approximately 6.7% positive (default), an imbalance that is handled explicitly

The raw dataset is not included in this repository. To run the notebook, download the data from the Kaggle competition page and place `cs-training.csv` in the project folder.

## Approach

The notebook follows a full modelling pipeline:

1. **Data preparation.** Missing-value handling, review of skew and outliers, and explicit treatment of the 6.7% class imbalance.
2. **Feature engineering.** Derived behavioural features including `DebtToIncome`, `DelinqSum`, `DebtRatio`, and an `IncomeMissing` indicator.
3. **Model construction.** Logistic Regression and Random Forest, both using `class_weight="balanced"` to address the imbalance.
4. **Hyperparameter optimisation.** Cross-validated search with GridSearchCV and RandomizedSearchCV, scored on ROC-AUC.
5. **Interpretation.** Logistic Regression coefficients and Random Forest permutation importance.
6. **Calibration.** Isotonic calibration to turn raw model scores into reliable probabilities.
7. **Cost-sensitive thresholding.** Selecting an operating point under a 10:1 cost ratio, where a false negative is treated as ten times more costly than a false positive.
8. **Stability testing.** Checking that performance holds across different random seeds.

## Results

| Model | ROC-AUC (test) |
|---|---|
| Random Forest | 0.863 |
| Logistic Regression | 0.856 |

* **Calibration.** Isotonic calibration improved the Brier score from 0.123 to 0.049 with no loss of ROC-AUC.
* **Cost-sensitive thresholding.** Moving from the default 0.5 threshold to a cost-aware operating point reduced missed defaulters (false negatives) from 2,526 to 902, a 64% reduction, at the expected cost of more false positives. This trade-off supports a deliberately risk-averse lending policy.

## Tech stack

Python, pandas, NumPy, scikit-learn (linear_model, ensemble, calibration, model_selection, metrics, inspection), imbalanced-learn, matplotlib, seaborn.

## Files

* `credit-default-prediction.ipynb` is the full analysis notebook, including code, figures, and results.

## Notes

This project was developed during the MSc Data Science programme at City St George's, University of London. It applies credit risk modelling, probability calibration, and cost-aware decision thresholds, connecting model outputs to the kind of cost structures used in IFRS 9 expected credit loss analysis.
