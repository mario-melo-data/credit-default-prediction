# Credit Default Prediction

Predicting serious borrower delinquency using Logistic Regression and Random Forest, with probability calibration and an illustrative cost-sensitive threshold analysis.

## Overview

This project builds and evaluates two credit-risk models on the [Give Me Some Credit](https://www.kaggle.com/c/GiveMeSomeCredit) dataset (150,000 borrower records, 10 predictors, 6.7% default rate). The aim is not only to rank borrowers by risk but to produce well-calibrated default probabilities and to examine how the decision threshold changes under an assumed cost of lending mistakes. In credit lending, failing to flag a defaulter is far more expensive than rejecting a good applicant, and the modelling here is built around that asymmetry.

## Data

* **Source:** Give Me Some Credit (Kaggle), file `cs-training.csv`
* **Size:** 150,000 records, 10 predictors
* **Target:** `SeriousDlqin2yrs`, where 1 indicates serious delinquency within two years
* **Class balance:** approximately 6.7% positive (default), an imbalance that is handled explicitly

The raw dataset is not included in this repository. To run the notebook, download the data from the Kaggle competition page and place `cs-training.csv` in the project folder.

## Approach

The notebook follows a full modelling pipeline:

1. **Data preparation.** Missing-value handling, review of skew and outliers, and explicit treatment of the 6.7% class imbalance.
2. **Feature engineering.** Engineered features: `DelinqSum` (total 30-59, 60-89 and 90+ day delinquencies), an `IncomeMissing` indicator and `DebtToIncome` (the original `DebtRatio` multiplied by `MonthlyIncome`, an estimate of monthly debt payments, set to 0 where either is missing).
3. **Model construction.** Logistic Regression and Random Forest, both using `class_weight="balanced"` to address the imbalance.
4. **Hyperparameter optimisation.** Cross-validated search with GridSearchCV and RandomizedSearchCV, scored on ROC-AUC.
5. **Interpretation.** Logistic Regression coefficients and Random Forest permutation importance.
6. **Calibration.** Isotonic calibration to turn raw model scores into reliable probabilities.
7. **Cost-sensitive thresholding.** Selecting an operating point under an assumed 10:1 cost ratio, where a false negative is treated as ten times more costly than a false positive.
8. **Stability testing.** Checking that performance holds across different random seeds.

## Results

| Model | ROC-AUC (test) |
|---|---|
| Random Forest | 0.863 |
| Logistic Regression | 0.856 |

* **Calibration.** Five-fold isotonic calibration reduced the test Brier score from 0.123 to 0.049. For context, a constant prediction of the 6.7% default rate would score about 0.06.
* **Cost-sensitive thresholding (illustrative).** Under an assumed 10:1 false-negative to false-positive cost ratio, the lowest tested threshold of 0.10 reduced false negatives from 2,526 to 902, while false positives rose from 306 to 6,043. The threshold was chosen using test-set labels at the edge of the searched range (the theoretical optimum for calibrated probabilities is 1/11, about 0.09), and the result is an exploratory trade-off rather than a deployable decision rule.

## Limitations

* DelinqSum and the three individual delinquency counts are almost perfectly correlated (|r| 0.987 to 0.997); the individual Logistic Regression coefficients on those counts are therefore not interpreted as independent risk effects.
* The Random Forest scores in the results table come from a 200-tree refit with the tuned depth and leaf settings; calibration and permutation importance use the tuned 400-tree model, which gives the same test ROC-AUC (0.863).

## Tech stack

Python, pandas, NumPy, scikit-learn (linear_model, ensemble, calibration, model_selection, metrics, inspection), imbalanced-learn, matplotlib, seaborn.

## Files

* `credit-default-prediction.ipynb` is the full analysis notebook, including code, figures, and results.

## Notes

This project was developed during the INM430 Principles of Data Science module of the MSc Data Science programme at City St George's, University of London. It applies credit risk modelling, probability calibration, and cost-aware decision thresholds.
