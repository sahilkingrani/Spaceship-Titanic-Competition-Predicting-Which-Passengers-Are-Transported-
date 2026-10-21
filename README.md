# Spaceship Titanic: Predicting Transported Passengers

End-to-end machine learning pipeline for the Kaggle competition [Spaceship Titanic](https://www.kaggle.com/competitions/spaceship-titanic): EDA, logic-based imputation, feature engineering, cross-validation, model comparison and XGBoost tuning.

**Final result:** 5-fold CV accuracy **0.8128** | Kaggle public leaderboard score **0.80289**

## Problem

In the year 2912, the passenger liner *Spaceship Titanic* hit a spacetime anomaly and some passengers were transported to an alternate dimension. Given passenger records (home planet, cabin, spending, age, etc.), the task is to predict whether each passenger was transported (`Transported`: True / False).

- **Task:** binary classification
- **Metric:** accuracy
- **Data:** 8,693 training rows, 4,277 test rows, 13 features
- **Target balance:** 50.4% True / 49.6% False (balanced, so accuracy is a reliable metric)

> The competition data is not included in this repository, as per the competition rules. Download it from the [competition page](https://www.kaggle.com/competitions/spaceship-titanic/data).

## Results

| Submission | Model | CV Accuracy (5-fold) | Kaggle Public Score |
|---|---|---|---|
| Version 1 | Random Forest + TotalSpend | 0.7996 | 0.79284 |
| Version 2 | XGBoost (max_depth=4, learning_rate=0.1) | 0.8128 | 0.80289 |

The CV-to-leaderboard gap is about 0.7% to 1.0%, which is consistent with mild optimism from picking the best configuration on the same CV folds, plus noise from the public leaderboard being scored on a subset of the test data.

## Approach

### 1. Exploratory Data Analysis

Key findings that shaped the modeling:

- **CryoSleep is the strongest signal:** 81.8% of passengers in cryosleep were transported vs 32.9% of those who were not.
- **Cryosleep passengers never spend money:** total spending is exactly 0 for every passenger in cryosleep. This became a rule for imputation.
- **Spending is heavily right-skewed:** about 42% of passengers spent nothing, while a few spent more than 20,000.
- **HomePlanet matters moderately:** Europa 65.9% transported vs Earth 42.4%.
- **Children were transported more:** 66.9% for ages 0-12 vs about 47% for young adults.
- **Missing values:** about 2% in every column (no single column is bad), so rows cannot simply be dropped.

### 2. Preprocessing

- **Logic-based imputation first:**
  - If `CryoSleep` is True, missing spending values are set to 0.
  - If a passenger has any spending and `CryoSleep` is missing, `CryoSleep` is set to False.
- **Statistical imputation for the rest:** mode for categorical columns, median for numeric columns. All statistics are computed on the **training set only** and applied to the test set, to avoid data leakage.
- **One-hot encoding** for `HomePlanet`, `Destination`, `Deck` and `Side` (`drop_first=True`).

### 3. Feature Engineering

| Feature | Source | Idea |
|---|---|---|
| `Deck`, `CabinNum`, `Side` | `Cabin` (format `deck/num/side`) | Split one compound field into three useful features |
| `GroupSize` | `PassengerId` (format `gggg_pp`) | How many people travel in the same group |
| `IsSolo` | `GroupSize` | Whether the passenger travels alone |
| `TotalSpend` | Five spending columns | Combined spending (added almost no gain for tree models) |

### 4. Validation Strategy

- Stratified 80/20 split for quick checks.
- **Stratified 5-fold cross-validation** as the main metric, because a single split can be lucky or unlucky (fold scores ranged from 0.78 to 0.81 for the same model).

### 5. Modeling

Models were improved one change at a time, with every experiment logged.

## Experiment Log

| # | Model | Features | Validation | CV Accuracy |
|---|---|---|---|---|
| 1 | Logistic Regression (default) | 23, unscaled | 80/20 split | 0.7918 |
| 2 | Logistic Regression (default) | 23, unscaled | 5-fold CV | 0.7919 (±0.0061) |
| 3 | Logistic Regression | 23, scaled | 80/20 split | 0.7936 |
| 4 | Random Forest (200 trees) | 23 | 5-fold CV | 0.7993 (±0.0107) |
| 5 | Random Forest (200 trees) | 24 (+TotalSpend) | 5-fold CV | 0.7996 (±0.0092) |
| 6 | XGBoost (depth 6, lr 0.3, defaults) | 24 | 5-fold CV | 0.8009 (±0.0096) |
| 7 | XGBoost (max_depth=3) | 24 | 5-fold CV | 0.8073 (±0.0061) |
| 8 | XGBoost (max_depth=4) | 24 | 5-fold CV | 0.8089 (±0.0084) |
| 9 | **XGBoost (max_depth=4, learning_rate=0.1)** | 24 | 5-fold CV | **0.8128 (±0.0060)** |

## Key Findings

- **Shallower trees generalized better.** Reducing XGBoost `max_depth` from 6 to 4 improved CV accuracy by about 0.8%, which suggests the default depth was slightly overfitting.
- **Tuning mattered more than switching models.** Logistic Regression, Random Forest and default XGBoost all landed between 0.79 and 0.80, while tuning added the largest gain.
- **A new feature is not automatically an improvement.** `TotalSpend` added only +0.0003 for Random Forest, since trees already capture the information from the individual spending columns.
- **Model errors are fairly balanced.** On a validation split, the Random Forest had precision/recall of 0.79/0.84 for the False class and 0.83/0.77 for the True class.
- **Default feature importance can mislead.** Random Forest importance ranked `CabinNum` and `Age` above `CryoSleep`, even though `CryoSleep` is the strongest signal in EDA. High-cardinality numeric features get more split opportunities, which inflates their importance scores.

## Challenges

- **Convergence warning in Logistic Regression** caused by features on very different scales (0/1 flags vs spending up to about 29,000). Fixed with `StandardScaler`, fitted on training data only.
- **Mixed data types after imputation:** `CryoSleep` and `VIP` became `object` columns because of missing values, and had to be converted back to boolean.
- **Matching train/test columns** after one-hot encoding, handled by reindexing the test set to the training column order before predicting.

## What I Would Improve Next

- Group-level features (for example, average spending or cryosleep share within a travel group).
- Family features from passenger surnames.
- Compare LightGBM and CatBoost, and try a simple ensemble.
- Use a systematic search (for example Optuna) with nested validation to reduce selection bias from tuning on the same folds.

## Skills Demonstrated

Exploratory data analysis, rule-based and statistical imputation, feature engineering, preventing data leakage, stratified cross-validation, model comparison (Logistic Regression, Random Forest, XGBoost), hyperparameter tuning, model interpretation (confusion matrix, precision/recall, feature importance), and honest comparison of validation vs leaderboard scores.

## Tech Stack

Python, pandas, NumPy, scikit-learn, XGBoost, Kaggle Notebooks.

## How to Run

1. Download the data from the [competition page](https://www.kaggle.com/competitions/spaceship-titanic/data).
2. Open the notebook (`.ipynb`) in this repository in Kaggle or Jupyter.
3. Update the data paths if needed and run all cells top to bottom.

---

*Built by Sahil as part of a hands-on Kaggle learning path in Data Science and Machine Learning.*
